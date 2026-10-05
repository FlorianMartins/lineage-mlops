"""Deploy a registry version: verify, export, hand to the backend, check parity.

Order matters. The version's signature, provenance and file hashes are verified
*before* anything is built from it; the merged export is hashed; the backend's result
is identified by digest; and a parity run on the held-out split must reproduce the
evaluated accuracy within a tolerance, because a format conversion (safetensors to
the backend's own format) is a change to the model that no one evaluated otherwise.
"""

from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from lineage.audit import AuditLog
from lineage.data import service as data_service
from lineage.errors import IntegrityError, LineageError, PolicyDenied
from lineage.evaluate.service import quality_metrics
from lineage.hashing import sha256_file
from lineage.registry.store import Registry
from lineage.serve import backends, drift
from lineage.supply import models
from lineage.workspace import Workspace

MODELFILE = '''FROM .
TEMPLATE """{template}"""
PARAMETER temperature 0
PARAMETER num_predict 24
PARAMETER stop "###"
'''


def deployments(ws: Workspace) -> Path:
    """Directory of deployment records."""
    return ws.state / "deployments"


def tag(name: str, number: int) -> str:
    """Backend model name for a registry version."""
    return f"lineage-{name}:v{number}"


def verified(ws: Workspace, ref: str) -> tuple[Registry, str, int]:
    """Resolve a version and refuse it unless files, signature and provenance verify."""
    registry = Registry(ws)
    name, number = registry.resolve(ref)
    check = registry.verify(number, name)
    if not check["ok"]:
        raise PolicyDenied(
            f"{name}:{number} does not verify; refusing to serve it", reasons=check["problems"]
        )
    return registry, name, number


def export(ws: Workspace, ref: str) -> Path:
    """Merge the verified adapter into the verified base model; write safetensors."""
    registry, name, number = verified(ws, ref)
    pin, snapshot = models.verify_snapshot(ws)
    manifest = registry.manifest(number, name)
    if manifest["base_model"] != pin.ref:
        raise PolicyDenied(f"{name}:{number} needs {manifest['base_model']}, pinned is {pin.ref}")
    target = ws.state / "exports" / f"{name}-v{number}"
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    from peft import PeftModel

    from lineage.train.lora import load_base

    model, tokenizer = load_base(snapshot)
    merged = PeftModel.from_pretrained(model, str(registry.path(number, name) / "adapter"))
    merged = merged.merge_and_unload()
    merged.save_pretrained(str(target), safe_serialization=True)
    tokenizer.save_pretrained(str(target))
    legacy_rope_keys(target / "config.json")
    task = data_service.task_of(ws)
    template = task.prompt_template.replace("{input}", "{{ .Prompt }}")
    (target / "Modelfile").write_text(MODELFILE.format(template=template))
    files = {p.name: sha256_file(p) for p in sorted(target.iterdir()) if p.is_file()}
    record = {
        "model": name,
        "version": number,
        "manifest_sha256": sha256_file(registry.path(number, name) / "manifest.json"),
        "base_model": pin.ref,
        "files": files,
    }
    (target / "export.json").write_text(json.dumps(record, indent=2) + "\n")
    AuditLog(ws.audit_log).append(
        "serve.exported",
        subjects={"model": f"{name}:{number}", "base_model": pin.ref},
        payload={"files": files, "manifest_sha256": record["manifest_sha256"]},
    )
    return target


def legacy_rope_keys(config_path: Path) -> None:
    """Write RoPE settings where serving backends still look for them.

    transformers 5 saves ``rope_parameters: {rope_theta: ...}`` and drops the top-level
    ``rope_theta``. Ollama 0.30's safetensors importer only reads the top-level key and
    silently falls back to the default base (10 000 instead of SmolLM2's 100 000): the
    model still answers, just worse (held-out exact match 0.25 instead of 0.40). The
    deployment parity check is what caught it. Writing both forms is harmless to
    transformers and fixes the import.
    """
    config = json.loads(config_path.read_text())
    params = config.get("rope_parameters") or {}
    if "rope_theta" not in config and "rope_theta" in params:
        config["rope_theta"] = params["rope_theta"]
    if params.get("rope_type", "default") != "default" and "rope_scaling" not in config:
        config["rope_scaling"] = {k: v for k, v in params.items() if k != "rope_theta"}
    config_path.write_text(json.dumps(config, indent=2) + "\n")


def check_export(path: Path) -> dict[str, Any]:
    """Re-hash an export directory against its record."""
    record: dict[str, Any] = json.loads((path / "export.json").read_text())
    for name, digest in record["files"].items():
        if sha256_file(path / name) != digest:
            raise IntegrityError(f"export {path.name}/{name} changed after it was built")
    return record


def deploy(ws: Workspace, ref: str, *, backend_name: str | None = None) -> dict[str, Any]:
    """Export, deploy to the backend, run the parity check, record the deployment."""
    settings = ws.section("serve")
    backend = backends.make(
        backend_name or str(settings.get("backend", "ollama")), settings.get("backend_url")
    )
    export_dir = export(ws, ref)
    record = check_export(export_dir)
    name, number = record["model"], record["version"]
    model = tag(name, number)
    deployed = backend.deploy(export_dir, model)

    # Parity: the served artefact must score like the evaluated one.
    task = data_service.task_of(ws)
    registry = Registry(ws)
    report = json.loads((registry.path(number, name) / "eval.json").read_text())
    dataset = data_service.store_of(ws).get(report["dataset"])
    heldout = list(dataset.split(str(ws.section("eval").get("heldout_split", "heldout"))).records)
    parity_n = int(settings.get("parity_examples", len(heldout)))
    sample = heldout[:parity_n]
    outputs = []
    for record_ in sample:
        prompt = record_.input if backend.name == "ollama" else task.prompt(record_.input)
        outputs.append(backend.complete(model, prompt, 24).text.split("###")[0].strip())
    served = quality_metrics(sample, outputs, task)["exact_match"]
    evaluated = float(report["results"]["quality"]["finetuned"]["exact_match"])
    tolerance = float(settings.get("parity_tolerance", 0.05))
    parity = {
        "examples": len(sample),
        "served_exact_match": served,
        "evaluated_exact_match": evaluated,
        "tolerance": tolerance,
        "passed": abs(served - evaluated) <= tolerance,
    }

    reference = drift.build_reference([r.input for r in heldout], [r.output for r in heldout], task)
    out = {
        "model": name,
        "version": number,
        "backend": backend.name,
        "backend_model": model,
        "backend_digest": deployed.get("digest"),
        "export_dir": str(export_dir),
        "export_files": record["files"],
        "manifest_sha256": record["manifest_sha256"],
        "parity": parity,
        "deployed_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "drift_reference": reference,
    }
    if "launch" in deployed:
        out["launch"] = deployed["launch"]
    log = AuditLog(ws.audit_log)
    subject = {"model": f"{name}:{number}"}
    if not parity["passed"]:
        log.append(
            "serve.deploy_denied",
            subjects=subject,
            payload={"backend": backend.name, "parity": parity},
        )
        raise PolicyDenied(
            f"served model scores {served:.3f} vs {evaluated:.3f} evaluated "
            f"(tolerance {tolerance}); the conversion changed the model",
        )
    deployments(ws).mkdir(parents=True, exist_ok=True)
    path = deployments(ws) / f"{name}-v{number}-{backend.name}.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    log.append(
        "serve.deployed",
        subjects=subject,
        payload={
            "backend": backend.name,
            "backend_model": model,
            "backend_digest": deployed.get("digest"),
            "parity": parity,
            "deployment_sha256": sha256_file(path),
        },
    )
    return out


def load_deployment(ws: Workspace, name: str, number: int, backend: str) -> dict[str, Any]:
    """The deployment record of a version on a backend."""
    path = deployments(ws) / f"{name}-v{number}-{backend}.json"
    if not path.exists():
        raise LineageError(f"{name}:{number} is not deployed on {backend} (run `lineage deploy`)")
    data: dict[str, Any] = json.loads(path.read_text())
    return data
