"""Registry operations: register, approve, promote, roll back.

Each operation gathers evidence by *re-verifying* it (never by trusting a flag written
earlier), hands it to the OPA policy, acts only on ``allow``, and logs the decision
either way. A denied promotion is an audit event too: "someone tried" is information.
"""

from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from typing import Any

from lineage import __version__
from lineage.audit import AuditLog
from lineage.data import service as data_service
from lineage.errors import LineageError, PolicyDenied
from lineage.evaluate import service as eval_service
from lineage.hashing import sha256_file
from lineage.registry.signing import Signer
from lineage.registry.store import Registry, ensure_integrity, write_json
from lineage.supply import mlbom, models
from lineage.train import service as train_service
from lineage.workspace import Workspace, current_actor


def _actor_of(log: AuditLog, event: str, run_id: str) -> str | None:
    for entry in log.entries():
        if entry.event == event and entry.subjects.get("run") == run_id:
            return entry.actor
    return None


def provenance(
    run: dict[str, Any], report: dict[str, Any], files: dict[str, str]
) -> dict[str, Any]:
    """SLSA v1 provenance predicate describing how the adapter was built."""
    repo, revision = run["base_model"].split("@", 1)
    code = run["environment"].get("code", {})
    dependencies = [
        {
            "uri": f"https://huggingface.co/{repo}/tree/{revision}",
            "digest": {"sha256": run["base_model_files"].get("model.safetensors", "")},
            "name": "base_model",
        },
        {
            "uri": f"lineage-dataset:{run['dataset']}",
            "digest": {"sha256": run["dataset"].removeprefix("ds-")},
            "name": "dataset",
        },
    ]
    if code.get("commit"):
        dependencies.append(
            {
                "uri": "git+https://github.com/FlorianMartins/lineage-mlops",
                "digest": {"gitCommit": code["commit"]},
                "name": "code",
            }
        )
    if run["environment"].get("lock_sha256"):
        dependencies.append(
            {
                "uri": f"file:{run['environment']['lock_file']}",
                "digest": {"sha256": run["environment"]["lock_sha256"]},
                "name": "dependency_lock",
            }
        )
    return {
        "buildDefinition": {
            "buildType": "https://github.com/FlorianMartins/lineage-mlops/lora-train@v1",
            "externalParameters": {"config": run["config"], "config_hash": run["config_hash"]},
            "internalParameters": {
                "python": run["environment"]["python"],
                "platform": run["environment"]["platform"],
                "packages": run["environment"]["packages"],
            },
            "resolvedDependencies": dependencies,
        },
        "runDetails": {
            "builder": {"id": f"lineage/{__version__}", "version": {"lineage": __version__}},
            "metadata": {"invocationId": run["run_id"], "startedOn": run["started"]},
            "byproducts": [
                {
                    "name": "adapter",
                    "digest": {"sha256": files.get("adapter/adapter_model.safetensors", "")},
                },
                {"name": "evaluation_report", "digest": {"sha256": report["report_hash"]}},
            ],
        },
    }


def model_card(name: str, number: int, run: dict[str, Any], report: dict[str, Any]) -> str:
    """Short human-readable card for the version."""
    quality = report["results"]["quality"]
    gates = report["gates"]
    lines = [
        f"# {name} — version {number}",
        "",
        f"- Base model: `{run['base_model']}`",
        f"- Dataset: `{run['dataset']}`",
        f"- Training run: `{run['run_id']}` (config hash `{run['config_hash'][:16]}`)",
        f"- Code: `{run['environment']['code'].get('commit')}`",
        "",
        "## Evaluation (held-out)",
        "",
        "| system | exact match | valid answers |",
        "|---|---:|---:|",
    ]
    for system in ("finetuned", "rag", "base"):
        lines.append(
            f"| {system} | {quality[system]['exact_match']:.3f} | {quality[system]['valid']:.3f} |"
        )
    lines += ["", "## Gates", ""]
    for gate, outcome in gates.items():
        mark = "passed" if outcome["passed"] else "FAILED: " + "; ".join(outcome["failures"])
        lines.append(f"- **{gate}**: {mark}")
    lines += [
        "",
        "## Intended use and limits",
        "",
        "Triage suggestions for IT support tickets, reviewed by a human. Not a security",
        "control: the red-team results above show which injections still succeed.",
        "",
    ]
    return "\n".join(lines)


def register(ws: Workspace, run_ref: str) -> int:
    """Copy a run into the registry as a new, signed candidate version."""
    registry = Registry(ws)
    run = train_service.load_run(ws, run_ref)
    digest = train_service.adapter_digest(run)
    report = eval_service.load_report(run.path, digest)
    for entry in registry.versions():
        if entry["adapter_digest"] == digest:
            raise LineageError(f"this adapter is already {registry.model_name}:{entry['version']}")
    pin, snapshot = models.verify_snapshot(ws)
    licence = json.loads((snapshot / "manifest.json").read_text())["licence"]
    task = data_service.task_of(ws)

    index = registry.index()
    number = max((int(n) for n in index["versions"]), default=0) + 1
    target = registry.path(number)
    target.mkdir(parents=True)
    try:
        shutil.copytree(run.adapter, target / "adapter")
        shutil.copy2(run.path / "run.json", target / "run.json")
        shutil.copy2(run.path / eval_service.REPORT, target / "eval.json")
        bom = mlbom.build(
            model_name=registry.model_name,
            version=number,
            adapter_digest=digest,
            adapter_files=run.record["adapter_files"],
            run=run.record,
            report=report,
            licence=licence,
            task={"name": task.name, "intended_use": ws.section("data").get("intended_use", "")},
        )
        write_json(target / "mlbom.cdx.json", bom)
        (target / "MODEL_CARD.md").write_text(
            model_card(registry.model_name, number, run.record, report)
        )
        files = {
            str(p.relative_to(target)): sha256_file(p)
            for p in sorted(target.rglob("*"))
            if p.is_file()
        }
        write_json(target / "provenance.json", provenance(run.record, report, files))
        files["provenance.json"] = sha256_file(target / "provenance.json")
        manifest = {
            "model": registry.model_name,
            "version": number,
            "adapter_digest": digest,
            "run": run.id,
            "dataset": run.record["dataset"],
            "base_model": pin.ref,
            "eval_report": report["report_hash"],
            "eval_passed": report["passed"],
            "licence": licence,
            "registered_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "registered_by": current_actor(),
            "audit_head": AuditLog(ws.audit_log).head(),
            "files": files,
        }
        write_json(target / "manifest.json", manifest)
        signer = Signer.from_workspace(ws)
        signer.sign(target / "manifest.json", target / "manifest.sig.bundle")
        signer.attest(
            target / "manifest.json", target / "provenance.json", target / "provenance.sig.bundle"
        )
    except BaseException:
        shutil.rmtree(target, ignore_errors=True)
        raise

    manifest_sha = sha256_file(target / "manifest.json")
    index["versions"][str(number)] = {
        "stage": "candidate",
        "adapter_digest": digest,
        "run": run.id,
        "manifest_sha256": manifest_sha,
        "approvals": [],
        "was_production": False,
    }
    index["history"].append(
        {
            "at": manifest["registered_at"],
            "version": number,
            "to": "candidate",
            "by": manifest["registered_by"],
        }
    )
    registry._save_index(index)
    AuditLog(ws.audit_log).append(
        "registry.registered",
        subjects={
            "model": f"{registry.model_name}:{number}",
            "run": run.id,
            "dataset": run.record["dataset"],
            "base_model": pin.ref,
        },
        payload={
            "manifest_sha256": manifest_sha,
            "adapter_digest": digest,
            "eval_report": report["report_hash"],
            "eval_passed": report["passed"],
            "mlbom_sha256": files["mlbom.cdx.json"],
            "signing": signer.describe(),
            "audit_head": manifest["audit_head"],
        },
    )
    return number


def approve(ws: Workspace, ref: str, reason: str) -> None:
    """Record a named approval of a version's exact manifest."""
    if len(reason.strip()) < 10:
        raise LineageError("an approval needs a real reason (10+ characters)")
    registry = Registry(ws)
    name, number = registry.resolve(ref)
    ensure_integrity(registry.check_files(number, name), f"{name}:{number}")
    index = registry.index(name)
    entry = index["versions"][str(number)]
    actor = current_actor()
    entry["approvals"].append(
        {"actor": actor, "manifest_sha256": entry["manifest_sha256"], "reason": reason.strip()}
    )
    registry._save_index(index)
    AuditLog(ws.audit_log).append(
        "registry.approved",
        subjects={"model": f"{name}:{number}", "run": entry["run"]},
        payload={"manifest_sha256": entry["manifest_sha256"], "reason": reason.strip()},
    )


def _request(
    ws: Workspace, registry: Registry, name: str, number: int, action: str, to_stage: str
) -> dict[str, Any]:
    """Build the policy input from freshly re-verified evidence."""
    entry = registry.index(name)["versions"][str(number)]
    base = registry.path(number, name)
    checks = registry.verify(number, name)
    manifest = registry.manifest(number, name)
    log = AuditLog(ws.audit_log)
    eval_info: dict[str, Any] = {"present": (base / "eval.json").exists()}
    if eval_info["present"]:
        report = json.loads((base / "eval.json").read_text())
        eval_info["intact"] = (
            sha256_file(base / "eval.json") == manifest["files"].get("eval.json")
            and report.get("adapter_digest") == entry["adapter_digest"]
            and report.get("report_hash") == manifest["eval_report"]
        )
        eval_info["gates"] = {k: bool(v["passed"]) for k, v in report["gates"].items()}
    licence = manifest["licence"]
    accepted = any(
        e.event == "model.licence_accepted"
        and e.subjects.get("base_model") == manifest["base_model"]
        for e in log.entries()
    )
    return {
        "action": action,
        "from_stage": entry["stage"],
        "to_stage": to_stage,
        "actor": current_actor(),
        "settings": {
            "required_approvals": registry.required_approvals,
            "separation_of_duties": registry.separation_of_duties,
        },
        "version": {
            "model": name,
            "number": number,
            "manifest_sha256": sha256_file(base / "manifest.json"),
            "manifest_intact": checks["manifest_intact"]
            and sha256_file(base / "manifest.json") == entry["manifest_sha256"],
            "signature_verified": checks["signature_verified"],
            "provenance_verified": checks["provenance_verified"],
            "eval": eval_info,
            "mlbom_present": "mlbom.cdx.json" in manifest["files"],
            "licence_verdict": licence.get("verdict"),
            "licence_accepted": accepted,
            "trained_by": _actor_of(log, "train.finished", entry["run"]),
            "registered_by": manifest["registered_by"],
            "approvals": entry["approvals"],
            "was_production": entry.get("was_production", False),
        },
        "_problems": checks["problems"],
    }


def _transition(
    ws: Workspace, ref: str, to_stage: str, action: str, reason: str | None
) -> dict[str, Any]:
    registry = Registry(ws)
    name, number = registry.resolve(ref)
    request = _request(ws, registry, name, number, action, to_stage)
    problems = request.pop("_problems")
    decision = registry.decide(request)
    subject = {
        "model": f"{name}:{number}",
        "run": registry.index(name)["versions"][str(number)]["run"],
    }
    log = AuditLog(ws.audit_log)
    if not decision["allow"]:
        log.append(
            f"registry.{action}_denied",
            subjects=subject,
            payload={
                "to": to_stage,
                "deny": decision["deny"],
                "policy_sha256": decision["policy_sha256"],
                "reason": reason,
            },
        )
        raise PolicyDenied(
            f"{action} of {name}:{number} to {to_stage} denied by policy",
            reasons=decision["deny"] + problems,
        )

    index = registry.index(name)
    previous = None
    if to_stage == "production":
        for other_number, other in index["versions"].items():
            if other["stage"] == "production":
                previous = int(other_number)
                other["stage"] = "rolled_back" if action == "rollback" else "archived"
    entry = index["versions"][str(number)]
    from_stage = entry["stage"]
    entry["stage"] = to_stage
    if to_stage == "production":
        entry["was_production"] = True
    at = datetime.now(UTC).isoformat(timespec="seconds")
    index["history"].append(
        {
            "at": at,
            "version": number,
            "from": from_stage,
            "to": to_stage,
            "by": current_actor(),
            "action": action,
            "reason": reason,
            "replaced": previous,
        }
    )
    registry._save_index(index)
    log.append(
        "registry.rolled_back" if action == "rollback" else "registry.promoted",
        subjects=subject,
        payload={
            "from": from_stage,
            "to": to_stage,
            "replaced": previous,
            "manifest_sha256": request["version"]["manifest_sha256"],
            "approvals": [a["actor"] for a in entry["approvals"]],
            "policy_sha256": decision["policy_sha256"],
            "reason": reason,
        },
    )
    return {
        "model": name,
        "version": number,
        "from": from_stage,
        "to": to_stage,
        "replaced": previous,
    }


def promote(ws: Workspace, ref: str, to_stage: str) -> dict[str, Any]:
    """Promote one stage forward, if the policy allows it."""
    if to_stage not in {"staging", "production"}:
        raise LineageError("promote --to staging|production")
    return _transition(ws, ref, to_stage, "promote", None)


def rollback(ws: Workspace, name: str | None, reason: str) -> dict[str, Any]:
    """Put the previous production version back in production."""
    if len(reason.strip()) < 10:
        raise LineageError("a rollback needs a reason (10+ characters)")
    registry = Registry(ws)
    model = name or registry.model_name
    index = registry.index(model)
    current = registry.production(model)
    if current is None:
        raise LineageError(f"{model} has nothing in production to roll back from")
    previous = [
        e
        for e in index["history"]
        if e.get("to") == "production"
        and e["version"] != current["version"]
        and index["versions"][str(e["version"])]["stage"] == "archived"
    ]
    if not previous:
        raise LineageError(f"{model} has no earlier production version to roll back to")
    target = previous[-1]["version"]
    return _transition(ws, f"{model}:{target}", "production", "rollback", reason.strip())
