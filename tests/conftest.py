from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from lineage.workspace import Workspace

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "triage"


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Workspace:
    """A fresh copy of the example workspace (config + data, no state)."""
    shutil.copy(EXAMPLE / "lineage.toml", tmp_path / "lineage.toml")
    shutil.copytree(EXAMPLE / "data", tmp_path / "data")
    monkeypatch.setenv("LINEAGE_ACTOR", "tester")
    monkeypatch.chdir(tmp_path)
    return Workspace.load(tmp_path)


@pytest.fixture
def clean_version(workspace: Workspace) -> str:
    from lineage.data import service

    version, _ = service.ingest(
        workspace,
        "triage",
        {
            "train": workspace.root / "data/train.jsonl",
            "heldout": workspace.root / "data/heldout.jsonl",
        },
    )
    return version


@pytest.fixture
def poisoned_version(workspace: Workspace) -> str:
    from lineage.data import service

    version, _ = service.ingest(
        workspace,
        "triage-poisoned",
        {
            "train": workspace.root / "data/poisoned.jsonl",
            "heldout": workspace.root / "data/heldout.jsonl",
        },
    )
    return version


# ---------------------------------------------------------------------------
# A tiny, offline causal LM and a fake hub serving it
# ---------------------------------------------------------------------------
class FakeHub:
    """Serves a local directory as if it were a hub repository."""

    def __init__(self, root: Path, licence: str | None = "apache-2.0") -> None:
        self.root = root
        self._licence = licence
        self.downloads: list[str] = []

    def files(self, repo, revision):
        from lineage.hashing import sha256_file
        from lineage.supply.models import RemoteFile

        return [
            RemoteFile(
                p.name, p.stat().st_size, sha256_file(p) if p.suffix == ".safetensors" else None
            )
            for p in sorted(self.root.iterdir())
            if p.is_file()
        ]

    def licence(self, repo, revision):
        return self._licence

    def download(self, repo, revision, name, dest):
        self.downloads.append(name)
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.root / name, dest / name)
        return dest / name


TINY_REVISION = "a" * 40


def replace_tables(text: str, name: str, block: str) -> str:
    """Drop ``[name]`` and its sub-tables from a TOML text, then append ``block``."""
    import re

    kept, skipping = [], False
    for line in text.splitlines():
        header = re.match(r"^\[([^\]]+)\]\s*$", line)
        if header:
            table = header.group(1)
            skipping = table == name or table.startswith(name + ".")
        if not skipping:
            kept.append(line)
    return "\n".join(kept).rstrip() + "\n\n" + block.rstrip() + "\n"


def pin_tiny(root: Path, hub: FakeHub, use: str = "commercial") -> Workspace:
    """Pin the tiny model in the workspace at ``root`` and set the intended use."""
    from lineage.supply.models import pin_block

    block, _ = pin_block(hub, "test/tiny-llama", TINY_REVISION, root / ".pin")
    path = root / "lineage.toml"
    text = replace_tables(path.read_text(), "base_model", block)
    text = replace_tables(text, "governance", f'[governance]\nintended_use = "{use}"')
    text = text.replace("epochs = 2", "epochs = 1").replace("threads = 4", "threads = 1")
    path.write_text(text)
    return Workspace.load(root)


@pytest.fixture(scope="session")
def tiny_model_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A 2-layer Llama with a word-level tokenizer trained on the example data."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    out = tmp_path_factory.mktemp("tiny-llama")
    texts = []
    for name in ("train.jsonl", "heldout.jsonl"):
        import json

        for line in (EXAMPLE / "data" / name).read_text().splitlines():
            row = json.loads(line)
            texts += [row["input"], row["output"]]
    texts.append("### Ticket ### Triage category priority :")
    tok = Tokenizer(models.WordLevel(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Sequence(
        [pre_tokenizers.WhitespaceSplit(), pre_tokenizers.Punctuation()]
    )
    tok.train_from_iterator(
        texts, trainers.WordLevelTrainer(special_tokens=["<unk>", "<pad>", "</s>"])
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tok, unk_token="<unk>", pad_token="<pad>", eos_token="</s>"
    )
    tokenizer.save_pretrained(out)
    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=len(tokenizer),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=None,
    )
    LlamaForCausalLM(config).save_pretrained(out, safe_serialization=True)
    return out


@pytest.fixture
def tiny_workspace(workspace: Workspace, tiny_model_dir: Path) -> Workspace:
    """The example workspace with the tiny model pinned and fetched."""
    from lineage.supply.models import fetch

    hub = FakeHub(tiny_model_dir)
    ws = pin_tiny(workspace.root, hub)
    fetch(ws, hub)
    return ws


# ---------------------------------------------------------------------------
# Registry and serving fixtures (need opa and cosign on PATH)
# ---------------------------------------------------------------------------
PERMISSIVE = """[gates.quality]
min_exact_match = 0.0
min_gain_over_base = -1.0
min_gain_over_rag = -1.0
max_regression = 1.0

[gates.privacy]
max_canary_exposure = 100.0
max_pii_leak_rate_over_base = 1.0

[gates.safety]
max_attack_success = 1.0
max_increase_over_base = 1.0
"""


@pytest.fixture
def promotable(tiny_workspace, clean_version, monkeypatch):
    """Tiny workspace with permissive gates, signing keys and two evaluated runs."""
    if not (shutil.which("opa") and shutil.which("cosign")):
        pytest.skip("needs opa and cosign on PATH")
    from lineage.data import service as data_service
    from lineage.evaluate import service as eval_service
    from lineage.registry.signing import init_keys
    from lineage.train import service as train_service

    monkeypatch.setenv("COSIGN_PASSWORD", "test")
    root = tiny_workspace.root
    text = (root / "lineage.toml").read_text()
    text = replace_tables(text, "eval", "[eval]\ncanary_candidates = 15\npii_probes = 5")
    for gate in ("gates.quality", "gates.privacy", "gates.safety"):
        text = replace_tables(text, gate, "")
    (root / "lineage.toml").write_text(text + "\n" + PERMISSIVE)
    shutil.copy(EXAMPLE / "redteam.jsonl", root / "redteam.jsonl")
    ws = Workspace.load(root)
    init_keys(ws)
    child, _ = data_service.plant_canaries(ws, clean_version, count=2, repeat=1, seed=5)
    data_service.run_validation(ws, child)
    runs = []
    for seed in (1, 2):
        monkeypatch.setenv("LINEAGE_ACTOR", "trainer")
        run = train_service.train(ws, child, seed=seed)
        eval_service.evaluate(ws, run.id)
        runs.append(run.id)
    monkeypatch.setenv("LINEAGE_ACTOR", "alice")
    return ws, runs


class TorchBackend:
    """Serves export directories with transformers, like a backend would."""

    name = "fake"

    def __init__(self, task):
        self.task = task
        self.models = {}
        self.digests = {}
        self.oracle = None  # prompt -> answer: a served model unlike the evaluated one

    def deploy(self, export_dir, model):
        from lineage.evaluate.predictor import Model

        self.models[model] = Model.load(export_dir, None, model, threads=1)
        self.digests[model] = "digest-" + model
        return {"model": model, "digest": self.digests[model]}

    def digest(self, model):
        return self.digests.get(model)

    def complete(self, model, prompt, max_tokens):
        from lineage.serve import backends

        if self.oracle is not None:
            return backends.Completion(self.oracle.get(prompt, "?"), 3, 2)
        text = self.models[model].generate([prompt], max_new_tokens=max_tokens, batch=1)[0]
        return backends.Completion(text, 3, 2)


@pytest.fixture
def served(promotable, monkeypatch):
    """Two versions registered, v1 in production, both deployed on the fake backend."""
    from lineage.data import service as data_service
    from lineage.registry import service as registry_service
    from lineage.serve import backends, deploy

    ws, runs = promotable
    backend = TorchBackend(data_service.task_of(ws))
    monkeypatch.setattr(backends, "make", lambda *_: backend)
    for run in runs:
        number = registry_service.register(ws, run)
        registry_service.promote(ws, f"ticket-triage:{number}", "staging")
        monkeypatch.setenv("LINEAGE_ACTOR", "bob")
        registry_service.approve(ws, f"ticket-triage:{number}", "reviewed the evaluation")
        monkeypatch.setenv("LINEAGE_ACTOR", "alice")
        deploy.deploy(ws, f"ticket-triage:{number}")
    registry_service.promote(ws, "ticket-triage:1", "production")
    return ws, backend
