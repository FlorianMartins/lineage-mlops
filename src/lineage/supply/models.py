"""Pinned base models.

A base model is identified by a repository *and an exact commit SHA*. Branch names and
tags move; a commit does not. Every file we download is listed in ``lineage.toml``
with its SHA-256, so a compromised mirror, a force-pushed repository or a corrupted
cache all fail the same way: the hash does not match and nothing is loaded.

``lineage model pin`` produces that block for you (it reads the hub's declared hashes
and refuses pickle files); ``lineage model fetch`` downloads, verifies, checks the
weight formats and the licence, and writes a manifest; ``verify_snapshot`` re-hashes
everything before each use.
"""

from __future__ import annotations

import json
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from lineage.audit import AuditLog
from lineage.errors import IntegrityError, LineageError, PolicyDenied
from lineage.hashing import sha256_file
from lineage.supply import licence, weights
from lineage.workspace import Workspace

SHA = re.compile(r"^[0-9a-f]{40}$")
# What loading a causal LM and its tokenizer actually reads. Training logs, results
# and alternative formats (ONNX, GGUF) stay on the hub.
NEEDED = re.compile(
    r"^(config|generation_config|tokenizer_config|special_tokens_map|added_tokens)\.json$"
    r"|^tokenizer\.(json|model)$|^vocab\.(json|txt)$|^merges\.txt$|^chat_template\.jinja$"
    r"|^model(-\d{5}-of-\d{5})?\.safetensors$|^model\.safetensors\.index\.json$"
)


@dataclass(frozen=True)
class RemoteFile:
    """A file as the hub describes it."""

    name: str
    size: int | None
    sha256: str | None  # declared by the hub for large (LFS) files


class Hub(Protocol):
    """Where base models come from. Tests substitute a local fake."""

    def files(self, repo: str, revision: str) -> list[RemoteFile]:
        """Files at an exact revision."""
        ...

    def licence(self, repo: str, revision: str) -> str | None:
        """Licence as declared by the model card."""
        ...

    def download(self, repo: str, revision: str, name: str, dest: Path) -> Path:
        """Download one file into ``dest`` and return its path."""
        ...


class HuggingFaceHub:
    """The Hugging Face Hub, always addressed by commit SHA."""

    def files(self, repo: str, revision: str) -> list[RemoteFile]:
        """List files with their LFS SHA-256 when the hub declares one."""
        from huggingface_hub import HfApi

        info = HfApi().model_info(repo, revision=revision, files_metadata=True)
        out = []
        for sibling in info.siblings or []:
            lfs = getattr(sibling, "lfs", None)
            sha = getattr(lfs, "sha256", None) if lfs else None
            out.append(RemoteFile(sibling.rfilename, getattr(sibling, "size", None), sha))
        return out

    def licence(self, repo: str, revision: str) -> str | None:
        """The ``license`` field of the model card."""
        from huggingface_hub import HfApi

        info = HfApi().model_info(repo, revision=revision)
        data = info.card_data.to_dict() if info.card_data else {}  # type: ignore[no-untyped-call]
        value = data.get("license")
        return str(value) if value else None

    def download(self, repo: str, revision: str, name: str, dest: Path) -> Path:
        """Download one file at the pinned revision."""
        from huggingface_hub import hf_hub_download

        return Path(hf_hub_download(repo, name, revision=revision, local_dir=dest))


@dataclass(frozen=True)
class ModelPin:
    """The ``[base_model]`` table."""

    repo: str
    revision: str
    licence: str
    files: dict[str, str]

    @classmethod
    def from_config(cls, section: dict[str, Any]) -> ModelPin:
        """Parse and validate the pin."""
        if not section:
            raise LineageError("lineage.toml has no [base_model] table")
        revision = str(section.get("revision", ""))
        if not SHA.match(revision):
            raise PolicyDenied(
                f"base model revision '{revision}' is not a 40-character commit SHA; "
                "branches and tags can move under you"
            )
        files = section.get("files")
        if not isinstance(files, dict) or not files:
            raise LineageError("[base_model.files] must map each file to its sha256")
        for name, digest in files.items():
            if not re.fullmatch(r"[0-9a-f]{64}", str(digest)):
                raise LineageError(f"[base_model.files] '{name}': not a sha256 digest")
        return cls(
            repo=str(section["repo"]),
            revision=revision,
            licence=str(section.get("licence", "")),
            files={str(k): str(v) for k, v in files.items()},
        )

    @property
    def ref(self) -> str:
        """``repo@revision``, the identity used in the audit log and the ML-BOM."""
        return f"{self.repo}@{self.revision}"

    @property
    def slug(self) -> str:
        """Directory name of the local snapshot."""
        return f"{self.repo.replace('/', '__')}@{self.revision[:12]}"


def snapshot_dir(ws: Workspace, pin: ModelPin) -> Path:
    """Where the verified snapshot lives."""
    return ws.models / pin.slug


def pin_block(hub: Hub, repo: str, revision: str, workdir: Path) -> tuple[str, list[str]]:
    """Produce the ``[base_model]`` TOML for a repo at a revision.

    Returns the TOML text and the files deliberately left out (pickles, ONNX, logs).
    Files the hub declares a SHA-256 for are cross-checked after download.
    """
    if not SHA.match(revision):
        raise PolicyDenied("pin an exact 40-character commit SHA, not a branch or tag")
    remote = hub.files(repo, revision)
    keep, skipped = [], []
    for item in remote:
        suffix = Path(item.name).suffix
        if NEEDED.match(item.name):
            keep.append(item)
        else:
            reason = "pickle" if suffix in weights.PICKLE_SUFFIXES else "not needed"
            skipped.append(f"{item.name} ({reason})")
    if not any(Path(i.name).suffix == ".safetensors" for i in keep):
        raise PolicyDenied(f"{repo}@{revision} has no safetensors weights; refusing to pin")
    lines = [
        "[base_model]",
        f'repo = "{repo}"',
        f'revision = "{revision}"',
        f'licence = "{hub.licence(repo, revision) or "UNKNOWN"}"',
        "",
        "[base_model.files]",
    ]
    for item in sorted(keep, key=lambda i: i.name):
        path = hub.download(repo, revision, item.name, workdir)
        digest = sha256_file(path)
        if item.sha256 and item.sha256 != digest:
            raise IntegrityError(f"{item.name}: hub declares {item.sha256}, got {digest}")
        lines.append(f'"{item.name}" = "{digest}"')
    return "\n".join(lines) + "\n", skipped


def fetch(ws: Workspace, hub: Hub, *, use: str | None = None) -> Path:
    """Download, verify and admit the pinned base model; return the snapshot path."""
    pin = ModelPin.from_config(ws.section("base_model"))
    use = use or str(ws.section("governance").get("intended_use", "internal"))
    target = snapshot_dir(ws, pin)
    log = AuditLog(ws.audit_log)

    reported = hub.licence(pin.repo, pin.revision)
    decision = licence.check(reported, use, declared=pin.licence)
    if decision.verdict == "deny":
        log.append(
            "model.rejected",
            subjects={"base_model": pin.ref},
            payload={"reason": "licence", "licence": decision.to_dict()},
        )
        raise PolicyDenied(f"licence check failed for {pin.ref}", reasons=decision.reasons)

    remote = {f.name: f for f in hub.files(pin.repo, pin.revision)}
    missing = sorted(set(pin.files) - set(remote))
    if missing:
        raise IntegrityError(f"{pin.ref} no longer has {missing}")

    ws.models.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".fetch-", dir=ws.models))
    try:
        for name, expected in pin.files.items():
            if Path(name).suffix in weights.PICKLE_SUFFIXES:
                raise PolicyDenied(f"{name} is a pickle-based file; only safetensors are loaded")
            path = hub.download(pin.repo, pin.revision, name, staging)
            if path != staging / name:
                shutil.copy2(path, staging / name)
            actual = sha256_file(staging / name)
            if actual != expected:
                log.append(
                    "model.rejected",
                    subjects={"base_model": pin.ref},
                    payload={
                        "reason": "hash_mismatch",
                        "file": name,
                        "expected": expected,
                        "actual": actual,
                    },
                )
                raise IntegrityError(f"{name}: expected sha256 {expected}, got {actual}")
        shutil.rmtree(staging / ".cache", ignore_errors=True)
        inspection = weights.inspect(staging)
        if not inspection.ok:
            raise PolicyDenied(
                f"{pin.ref} failed the weight policy",
                reasons=inspection.problems
                + [f"pickle file: {p}" for p in inspection.pickles]
                + [f"unexpected file: {p}" for p in inspection.unknown],
            )
        manifest = {
            "repo": pin.repo,
            "revision": pin.revision,
            "files": pin.files,
            "licence": decision.to_dict(),
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        if target.exists():
            shutil.rmtree(target)
        staging.rename(target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    log.append(
        "model.fetched",
        subjects={"base_model": pin.ref},
        payload={"files": pin.files, "licence": decision.to_dict(), "use": use},
    )
    return target


def verify_snapshot(ws: Workspace) -> tuple[ModelPin, Path]:
    """Re-hash the local snapshot against the pin; refuse anything that drifted."""
    pin = ModelPin.from_config(ws.section("base_model"))
    target = snapshot_dir(ws, pin)
    if not (target / "manifest.json").exists():
        raise PolicyDenied(f"{pin.ref} is not fetched (run `lineage model fetch`)")
    for name, expected in pin.files.items():
        path = target / name
        if not path.exists() or sha256_file(path) != expected:
            raise IntegrityError(f"{pin.slug}/{name} does not match its pinned sha256")
    inspection = weights.inspect(target)
    extra = set(
        inspection.safetensors + inspection.data + inspection.pickles + inspection.unknown
    ) - set(pin.files)
    if extra or not inspection.ok:
        raise PolicyDenied(
            f"{pin.slug} contains files outside the pin or failing the weight policy",
            reasons=sorted(extra) + inspection.problems,
        )
    manifest = json.loads((target / "manifest.json").read_text())
    if manifest["licence"]["verdict"] == "conditional" and not _conditions_accepted(ws, pin):
        raise PolicyDenied(
            f"{pin.ref} is licensed under conditions nobody accepted yet "
            "(run `lineage model accept-licence`)",
            reasons=manifest["licence"]["conditions"],
        )
    return pin, target


def _conditions_accepted(ws: Workspace, pin: ModelPin) -> bool:
    return any(
        e.event == "model.licence_accepted" and e.subjects.get("base_model") == pin.ref
        for e in AuditLog(ws.audit_log).entries()
    )


def accept_licence(ws: Workspace, reason: str) -> None:
    """Record that a named person accepts the licence conditions of the pinned model."""
    pin = ModelPin.from_config(ws.section("base_model"))
    manifest_path = snapshot_dir(ws, pin) / "manifest.json"
    if not manifest_path.exists():
        raise PolicyDenied(f"{pin.ref} is not fetched yet")
    conditions = json.loads(manifest_path.read_text())["licence"]["conditions"]
    AuditLog(ws.audit_log).append(
        "model.licence_accepted",
        subjects={"base_model": pin.ref},
        payload={"conditions": conditions, "reason": reason},
    )
