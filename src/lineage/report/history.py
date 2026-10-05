"""Reconstruct the full history of a model version from the audit log.

Starting from ``name:version``, the log's subject links are followed outwards: the
version names its training run; the run names its dataset version and base model;
a dataset names its parent (canaries, derivations) up to the raw ingestion. Every
entry that mentions any of those identifiers is part of the story. The result is one
ordered timeline (data, code, run, evaluation, approval, signature, deployment,
serving) with the hash of each entry, so any line can be checked against the chain.
"""

from __future__ import annotations

from typing import Any

from lineage.audit import AuditLog, Entry
from lineage.errors import LineageError
from lineage.registry.store import Registry
from lineage.workspace import Workspace

STAGE = {
    "data": "data",
    "model": "base model",
    "train": "training",
    "eval": "evaluation",
    "registry": "registry",
    "serve": "serving",
    "monitor": "monitoring",
    "cloud": "cloud",
}


def lineage_ids(ws: Workspace, ref: str) -> dict[str, Any]:
    """The identifiers that make up a version: itself, run, datasets, base model."""
    registry = Registry(ws)
    name, number = registry.resolve(ref)
    manifest = registry.manifest(number, name)
    datasets = [manifest["dataset"]]
    entries = list(AuditLog(ws.audit_log).entries())
    # Walk dataset parents through the log (canary planting records the parent).
    frontier = manifest["dataset"]
    while True:
        parent = next(
            (
                e.subjects["parent_dataset"]
                for e in entries
                if e.subjects.get("dataset") == frontier and "parent_dataset" in e.subjects
            ),
            None,
        )
        if not parent or parent in datasets:
            break
        datasets.append(parent)
        frontier = parent
    return {
        "model": f"{name}:{number}",
        "run": manifest["run"],
        "datasets": datasets,
        "base_model": manifest["base_model"],
        "manifest": manifest,
    }


def history(ws: Workspace, ref: str) -> dict[str, Any]:
    """Timeline of every audit entry about this version and what it was made from."""
    ids = lineage_ids(ws, ref)
    model, run, base = ids["model"], ids["run"], ids["base_model"]
    datasets = set(ids["datasets"])
    plans: set[str] = set()

    def relevant(entry: Entry) -> bool:
        subjects = entry.subjects
        # About another version or another run: not part of this story, even if it
        # shares the dataset or the base model.
        if subjects.get("model", model) != model or subjects.get("run", run) != run:
            return False
        if subjects.get("model") == model or subjects.get("run") == run:
            return True
        if subjects.get("dataset") in datasets or subjects.get("plan") in plans:
            return True
        return subjects.get("base_model") == base and entry.event.startswith("model.")

    timeline: list[Entry] = []
    for entry in AuditLog(ws.audit_log).entries():
        if relevant(entry):
            timeline.append(entry)
            if "plan" in entry.subjects:
                plans.add(entry.subjects["plan"])
    if not timeline:
        raise LineageError(f"nothing in the audit log about {ids['model']}")
    return {
        "model": ids["model"],
        "run": ids["run"],
        "datasets": ids["datasets"],
        "base_model": ids["base_model"],
        "events": [
            {
                "seq": e.seq,
                "ts": e.ts,
                "stage": STAGE.get(e.event.split(".")[0], e.event.split(".")[0]),
                "event": e.event,
                "actor": e.actor,
                "summary": summarise(e),
                "hash": e.hash,
            }
            for e in timeline
        ],
    }


def summarise(entry: Entry) -> str:
    """One readable line per event type."""
    p = entry.payload
    match entry.event:
        case "data.ingested":
            return f"ingested {p.get('name')} {p.get('splits')}"
        case "data.validated":
            s = p.get("summary", {})
            return f"validated: {s.get('high')} high, {s.get('warning')} warning findings"
        case "data.findings_acknowledged":
            return f"high findings accepted: {p.get('reason')}"
        case "data.canaries_planted":
            return f"{p.get('count')} canaries planted (x{p.get('repeat')})"
        case "model.fetched":
            lic = p.get("licence", {})
            return f"base model verified, licence {lic.get('licence')} -> {lic.get('verdict')}"
        case "model.licence_accepted":
            return f"licence conditions accepted: {p.get('reason')}"
        case "train.started":
            code = p.get("code", {})
            return (
                f"training started, config {str(p.get('config_hash'))[:12]}, "
                f"commit {str(code.get('commit'))[:10]}{' (dirty)' if code.get('dirty') else ''}"
            )
        case "train.finished":
            m = p.get("metrics", {})
            source = f" ({p['source']})" if p.get("source") else ""
            return f"training finished{source}: loss {m.get('first_loss')} -> {m.get('final_loss')}"
        case "eval.completed":
            gates = p.get("gates", {})
            verdict = ", ".join(
                f"{k} {'pass' if v['passed'] else 'FAIL'}" for k, v in gates.items()
            )
            return f"gates: {verdict}; exact match {p.get('exact_match', {}).get('finetuned')}"
        case "registry.registered":
            return (
                f"registered, signed ({p.get('signing', {}).get('mode')}), "
                f"manifest {str(p.get('manifest_sha256'))[:12]}"
            )
        case "registry.approved":
            return f"approved: {p.get('reason')}"
        case "registry.promoted":
            return f"promoted {p.get('from')} -> {p.get('to')}" + (
                f" (replaced v{p['replaced']})" if p.get("replaced") else ""
            )
        case "registry.rolled_back":
            return f"rollback to production (v{p.get('replaced')} pulled): {p.get('reason')}"
        case "registry.promote_denied" | "registry.rollback_denied":
            return "DENIED by policy: " + "; ".join(p.get("deny", []))
        case "serve.exported":
            return "exported merged safetensors"
        case "serve.deployed":
            parity = p.get("parity", {})
            return (
                f"deployed on {p.get('backend')} digest {str(p.get('backend_digest'))[:12]}, "
                f"parity {parity.get('served_exact_match')} "
                f"vs {parity.get('evaluated_exact_match')}"
            )
        case "serve.deploy_denied":
            return "deployment refused: parity check failed"
        case "serve.live":
            return f"now serving on {p.get('backend')}"
        case "monitor.drift_alert":
            return "drift: " + ", ".join(b["signal"] for b in p.get("breaches", []))
        case "cloud.planned":
            return f"cloud plan to {p.get('provider')} {p.get('region')}"
        case "cloud.consented":
            return f"consent by {p.get('by')} until {p.get('expires')}"
        case "cloud.executed":
            return "plan executed (data sent)"
        case _:
            return ", ".join(f"{k}={v}" for k, v in list(p.items())[:3])[:100]
