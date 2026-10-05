"""Deep verification: the audit log, and everything on disk against it.

A valid hash chain proves the log was not edited; it does not prove the rest of the
state agrees with it. ``verify_all`` cross-checks both directions:

* **chain**        every hash and link (``audit verify``), plus optional external anchors
                   (heads published elsewhere must still be in the chain)
* **datasets**     every stored version re-hashes and was logged when ingested;
                   validation reports are intact and logged with the same hash
* **runs**         adapters match their run record and the ``train.finished`` entry
* **registry**     files, signatures and provenance verify; the audit head embedded in
                   each signed manifest is in the chain (a rewritten or truncated log
                   fails here); the stage of every version in ``index.json`` is the one
                   the last logged transition gave it (a hand-edited index that puts a
                   version in production without the policy fails here)
* **deployments**  each record matches the hash logged when it was deployed

The result lists every check with its outcome, so a failure says what disagreed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from lineage.audit import AuditLog, Entry
from lineage.data.store import DatasetStore
from lineage.errors import LineageError
from lineage.hashing import sha256_file, sha256_json
from lineage.workspace import Workspace


@dataclass
class Check:
    """One verified fact."""

    area: str
    subject: str
    ok: bool
    detail: str = ""


@dataclass
class Verification:
    """All checks and the chain summary."""

    checks: list[Check] = field(default_factory=list)
    entries: int = 0
    head: str = ""

    @property
    def ok(self) -> bool:
        """True when every check passed."""
        return all(c.ok for c in self.checks)

    def add(self, area: str, subject: str, ok: bool, detail: str = "") -> None:
        """Record a check."""
        self.checks.append(Check(area, subject, ok, detail))

    def to_dict(self) -> dict[str, Any]:
        """Report form."""
        return {
            "ok": self.ok,
            "entries": self.entries,
            "head": self.head,
            "checks": [c.__dict__ for c in self.checks],
            "failed": [c.__dict__ for c in self.checks if not c.ok],
        }


def verify_all(ws: Workspace, anchors: list[str] | None = None) -> Verification:
    """Run every check."""
    result = Verification()
    log = AuditLog(ws.audit_log)
    chain = log.verify()
    result.entries, result.head = chain.entries, chain.head
    result.add(
        "chain",
        "audit log",
        chain.ok,
        f"{chain.entries} entries" if chain.ok else "; ".join(chain.errors[:5]),
    )
    entries = list(log.entries())
    hashes = {e.hash for e in entries}
    for anchor in anchors or []:
        result.add(
            "chain",
            f"anchor {anchor[:12]}",
            anchor in hashes,
            "" if anchor in hashes else "a published head is no longer in the chain",
        )
    _datasets(ws, entries, result)
    _runs(ws, entries, result)
    _registry(ws, entries, hashes, result)
    _deployments(ws, entries, result)
    return result


def _last(entries: list[Entry], event: str, **subjects: str) -> Entry | None:
    found = None
    for entry in entries:
        if entry.event == event and all(entry.subjects.get(k) == v for k, v in subjects.items()):
            found = entry
    return found


def _datasets(ws: Workspace, entries: list[Entry], result: Verification) -> None:
    store = DatasetStore(ws.datasets)
    for version in store.versions():
        try:
            store.get(version)
            ok, detail = True, ""
        except LineageError as exc:
            ok, detail = False, str(exc)
        result.add("datasets", version[:15], ok, detail)
        created = _last(entries, "data.ingested", dataset=version) or _last(
            entries, "data.canaries_planted", dataset=version
        )
        result.add(
            "datasets",
            f"{version[:15]} logged",
            created is not None,
            "" if created else "stored but never logged as ingested or derived",
        )
        report_path = store.path(version) / "validation.json"
        if report_path.exists():
            report = json.loads(report_path.read_text())
            body = {k: v for k, v in report.items() if k != "report_hash"}
            intact = sha256_json(body) == report.get("report_hash")
            logged = _last(entries, "data.validated", dataset=version)
            matches = logged is not None and logged.payload.get("report_hash") == report.get(
                "report_hash"
            )
            result.add(
                "datasets",
                f"{version[:15]} validation report",
                intact and matches,
                "" if intact and matches else "report edited or not the one logged",
            )


def _runs(ws: Workspace, entries: list[Entry], result: Verification) -> None:
    if not ws.runs.exists():
        return
    for path in sorted(p for p in ws.runs.iterdir() if (p / "run.json").exists()):
        record = json.loads((path / "run.json").read_text())
        problems = [
            name
            for name, digest in record["adapter_files"].items()
            if not (path / "adapter" / name).exists()
            or sha256_file(path / "adapter" / name) != digest
        ]
        finished = _last(entries, "train.finished", run=path.name)
        if finished is None:
            problems.append("no train.finished entry")
        elif finished.payload.get("adapter_files") != record["adapter_files"]:
            problems.append("adapter hashes differ from the logged ones")
        result.add("runs", path.name, not problems, "; ".join(problems))


def _registry(ws: Workspace, entries: list[Entry], hashes: set[str], result: Verification) -> None:
    from lineage.registry.store import Registry

    if not ws.registry.exists():
        return
    registry = Registry(ws)
    logged_stages = replay_stages(entries)
    for model_dir in sorted(p for p in ws.registry.iterdir() if (p / "index.json").exists()):
        name = model_dir.name
        index = registry.index(name)
        for number, entry in sorted(index["versions"].items(), key=lambda kv: int(kv[0])):
            subject = f"{name}:{number}"
            check = registry.verify(int(number), name)
            result.add(
                "registry", f"{subject} files+signatures", check["ok"], "; ".join(check["problems"])
            )
            manifest = registry.manifest(int(number), name)
            anchored = manifest.get("audit_head") in hashes or manifest.get("audit_head") == (
                "0" * 64
            )
            result.add(
                "registry",
                f"{subject} audit anchor",
                anchored,
                ""
                if anchored
                else "the head signed into the manifest is not in the "
                "chain: the log was rewritten or truncated",
            )
            expected = logged_stages.get(subject)
            result.add(
                "registry",
                f"{subject} stage",
                expected == entry["stage"],
                ""
                if expected == entry["stage"]
                else f"index says {entry['stage']}, the log says {expected}",
            )


def replay_stages(entries: list[Entry]) -> dict[str, str]:
    """The stage of every version according to the log alone, replayed in order."""
    stages: dict[str, str] = {}
    for entry in entries:
        subject = entry.subjects.get("model", "")
        if entry.event == "registry.registered":
            stages[subject] = "candidate"
        elif entry.event in {"registry.promoted", "registry.rolled_back"}:
            stages[subject] = str(entry.payload.get("to"))
            replaced = entry.payload.get("replaced")
            if replaced is not None:
                name = subject.split(":")[0]
                left = "rolled_back" if entry.event == "registry.rolled_back" else "archived"
                stages[f"{name}:{replaced}"] = left
    return stages


def _deployments(ws: Workspace, entries: list[Entry], result: Verification) -> None:
    directory = ws.state / "deployments"
    if not directory.exists():
        return
    for path in sorted(directory.glob("*.json")):
        record = json.loads(path.read_text())
        subject = f"{record['model']}:{record['version']}"
        logged = None
        for entry in entries:
            if (
                entry.event == "serve.deployed"
                and entry.subjects.get("model") == subject
                and entry.payload.get("backend") == record["backend"]
            ):
                logged = entry
        ok = logged is not None and logged.payload.get("deployment_sha256") == sha256_file(path)
        result.add(
            "deployments",
            f"{subject} on {record['backend']}",
            ok,
            "" if ok else "record differs from the one logged at deployment",
        )
