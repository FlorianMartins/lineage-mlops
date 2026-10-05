"""Dataset operations as the CLI and the pipeline use them: store + checks + audit."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from lineage.audit import AuditLog
from lineage.data import canary, card, validate
from lineage.data.store import DatasetStore, DatasetVersion, Split, read_records
from lineage.errors import LineageError, PolicyDenied
from lineage.hashing import sha256_file, sha256_json
from lineage.task import TaskSpec
from lineage.workspace import Workspace


def task_of(ws: Workspace) -> TaskSpec:
    """The task contract declared in ``lineage.toml``."""
    return TaskSpec.from_config(ws.section("task"))


def store_of(ws: Workspace) -> DatasetStore:
    """The workspace dataset store."""
    return DatasetStore(ws.datasets)


def ingest(ws: Workspace, name: str, splits: dict[str, Path]) -> tuple[str, bool]:
    """Read raw files into a new immutable version and log it."""
    if not splits:
        raise LineageError("give at least one split, e.g. train=data/train.jsonl")
    fields = ws.section("data")
    parts = []
    sources = {}
    for split_name, path in splits.items():
        records = read_records(
            path,
            str(fields.get("input_field", "input")),
            str(fields.get("output_field", "output")),
        )
        parts.append(Split(split_name, tuple(records)))
        sources[split_name] = {"path": str(path), "sha256": sha256_file(path)}
    dataset = DatasetVersion(name=name, splits=tuple(parts), provenance={"sources": sources})
    version, created = store_of(ws).put(dataset)
    if created:
        AuditLog(ws.audit_log).append(
            "data.ingested",
            subjects={"dataset": version},
            payload={
                "name": name,
                "splits": {s.name: len(s.records) for s in parts},
                "sources": sources,
            },
        )
    return version, created


def canary_hashes(ws: Workspace, version: str) -> set[str]:
    """Hashes of the canaries planted in ``version`` (or its ancestors)."""
    store = store_of(ws)
    found: set[str] = set()
    current: str | None = version
    while current:
        path = ws.canaries / f"{current}.json"
        if path.exists():
            found |= set(json.loads(path.read_text(encoding="utf-8"))["record_hashes"])
        current = store.manifest(current).get("parent")
    return found


def run_validation(ws: Workspace, version: str) -> dict[str, Any]:
    """Validate, store the report and the data card, log both hashes."""
    store = store_of(ws)
    dataset = store.get(version)
    task = task_of(ws)
    settings = ws.section("data")
    excluded = canary_hashes(ws, version)
    report = validate.validate(
        dataset,
        task,
        pii_severity=str(settings.get("pii_severity", "warning")),
        exclude=excluded,
    )
    store.write_artifact(version, "validation.json", json.dumps(report, indent=2) + "\n")
    card_text = card.render(
        dataset, task, report, ws.section("data"), canaries=report["excluded_canary_records"]
    )
    card_path = store.write_artifact(version, "DATA_CARD.md", card_text)
    AuditLog(ws.audit_log).append(
        "data.validated",
        subjects={"dataset": version},
        payload={
            "report_hash": report["report_hash"],
            "summary": report["summary"],
            "data_card_sha256": sha256_file(card_path),
        },
    )
    return report


def load_report(ws: Workspace, version: str) -> dict[str, Any]:
    """The stored validation report, re-checked against its own hash."""
    path = store_of(ws).path(version) / "validation.json"
    if not path.exists():
        raise PolicyDenied(f"{version} has never been validated (run `lineage data validate`)")
    report: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    body = {k: v for k, v in report.items() if k != "report_hash"}
    if sha256_json(body) != report.get("report_hash"):
        raise PolicyDenied(f"{version}: validation report was edited after it was produced")
    return report


def acknowledge(ws: Workspace, version: str, reason: str) -> str:
    """Record that a named person reviewed the open findings and accepts them."""
    if len(reason.strip()) < 10:
        raise LineageError("an acknowledgement needs a real reason (10+ characters)")
    report = load_report(ws, version)
    AuditLog(ws.audit_log).append(
        "data.findings_acknowledged",
        subjects={"dataset": version},
        payload={
            "report_hash": report["report_hash"],
            "high": report["summary"]["high"],
            "reason": reason.strip(),
        },
    )
    return str(report["report_hash"])


def require_trainable(ws: Workspace, version: str) -> dict[str, Any]:
    """Refuse a dataset that was not validated, or whose high findings nobody accepted."""
    report = load_report(ws, version)
    blocking = validate.blocking(report)
    if not blocking:
        return report
    acknowledged = any(
        e.event == "data.findings_acknowledged"
        and e.subjects.get("dataset") == version
        and e.payload.get("report_hash") == report["report_hash"]
        for e in AuditLog(ws.audit_log).entries()
    )
    if not acknowledged:
        checks = sorted({f["check"] for f in blocking})
        raise PolicyDenied(
            f"{version} has {len(blocking)} high finding(s) ({', '.join(checks)}); "
            "review them and run `lineage data acknowledge` to accept them explicitly",
            reasons=[f"{f['check']}: {f['detail']}" for f in blocking],
        )
    return report


def plant_canaries(
    ws: Workspace, version: str, *, count: int, repeat: int, seed: int | None
) -> tuple[str, int]:
    """Create a child version with canaries; keep the secrets outside the dataset."""
    store = store_of(ws)
    task = task_of(ws)
    child, canaries = canary.plant(store.get(version), task, count=count, repeat=repeat, seed=seed)
    child_version, _ = store.put(child)
    ws.canaries.mkdir(parents=True, exist_ok=True)
    secrets_path = ws.canaries / f"{child_version}.json"
    secrets_path.write_text(
        json.dumps(
            {
                "dataset": child_version,
                "parent": version,
                "repeat": repeat,
                "canaries": [c.to_dict() for c in canaries],
                "record_hashes": sorted(canary.hashes(canaries, task)),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    secrets_path.chmod(0o600)
    AuditLog(ws.audit_log).append(
        "data.canaries_planted",
        subjects={"dataset": child_version, "parent_dataset": version},
        payload={
            "count": count,
            "repeat": repeat,
            "canary_file_sha256": sha256_file(secrets_path),
        },
    )
    return child_version, count


def load_canaries(ws: Workspace, version: str) -> list[canary.Canary]:
    """Canaries planted in ``version`` or any ancestor."""
    store = store_of(ws)
    found: list[canary.Canary] = []
    current: str | None = version
    while current:
        path = ws.canaries / f"{current}.json"
        if path.exists():
            raw = json.loads(path.read_text(encoding="utf-8"))
            found += [canary.Canary(c["prefix"], c["secret"], c["labels"]) for c in raw["canaries"]]
        current = store.manifest(current).get("parent")
    return found
