"""Command-line interface.

Exit codes: 0 success, 1 a gate or policy said no, 2 usage or integrity error.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from lineage import __version__
from lineage.audit import AuditLog
from lineage.errors import LineageError, PolicyDenied
from lineage.workspace import Workspace

Handler = Callable[[argparse.Namespace, Workspace], int]
_COMMANDS: dict[str, Handler] = {}


def command(name: str) -> Callable[[Handler], Handler]:
    """Register a handler for ``lineage <group> <name>``."""

    def register(func: Handler) -> Handler:
        _COMMANDS[name] = func
        return func

    return register


def emit(args: argparse.Namespace, human: str, data: Any) -> None:
    """Print JSON with ``--json``, the human text otherwise."""
    if getattr(args, "json", False):
        print(json.dumps(data, indent=2, ensure_ascii=False, default=str))
    else:
        print(human)


def _splits(values: Sequence[str]) -> dict[str, Path]:
    splits = {}
    for value in values:
        name, sep, path = value.partition("=")
        if not sep or not name or not path:
            raise LineageError(f"split '{value}' must look like name=path")
        splits[name] = Path(path)
    return splits


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
@command("data.ingest")
def _data_ingest(args: argparse.Namespace, ws: Workspace) -> int:
    from lineage.data import service

    version, created = service.ingest(ws, args.name, _splits(args.splits))
    state = "stored" if created else "already stored (same content)"
    emit(args, f"{version}  {state}", {"version": version, "created": created})
    return 0


@command("data.validate")
def _data_validate(args: argparse.Namespace, ws: Workspace) -> int:
    from lineage.data import service

    version = service.store_of(ws).resolve(args.version)
    report = service.run_validation(ws, version)
    summary = report["summary"]
    lines = [
        f"{version}",
        f"  high: {summary['high']}   warning: {summary['warning']}",
    ]
    for finding in report["findings"]:
        lines.append(
            f"  [{finding['severity']:>7}] {finding['split']}/{finding['check']}: "
            f"{finding['detail']}"
        )
    lines.append(f"  report {report['report_hash']}")
    lines.append(f"  data card: {service.store_of(ws).path(version) / 'DATA_CARD.md'}")
    emit(args, "\n".join(lines), report)
    if args.fail_on_high and summary["high"]:
        return 1
    return 0


@command("data.card")
def _data_card(args: argparse.Namespace, ws: Workspace) -> int:
    from lineage.data import service

    version = service.store_of(ws).resolve(args.version)
    path = service.store_of(ws).path(version) / "DATA_CARD.md"
    if not path.exists():
        raise LineageError(f"no data card yet; run `lineage data validate {version}`")
    print(path.read_text(encoding="utf-8"))
    return 0


@command("data.acknowledge")
def _data_ack(args: argparse.Namespace, ws: Workspace) -> int:
    from lineage.data import service

    version = service.store_of(ws).resolve(args.version)
    digest = service.acknowledge(ws, version, args.reason)
    emit(args, f"acknowledged report {digest} for {version}", {"report_hash": digest})
    return 0


@command("data.plant-canaries")
def _data_canaries(args: argparse.Namespace, ws: Workspace) -> int:
    from lineage.data import service

    version = service.store_of(ws).resolve(args.version)
    child, count = service.plant_canaries(
        ws, version, count=args.count, repeat=args.repeat, seed=args.seed
    )
    emit(
        args,
        f"{child}  ({count} canaries x{args.repeat} planted into {version}; "
        "secrets kept in .lineage/canaries/)",
        {"version": child, "parent": version, "count": count, "repeat": args.repeat},
    )
    return 0


@command("data.list")
def _data_list(args: argparse.Namespace, ws: Workspace) -> int:
    from lineage.data import service

    store = service.store_of(ws)
    rows = []
    for version in store.versions():
        manifest = store.manifest(version)
        splits = ", ".join(f"{k}={v['records']}" for k, v in manifest["splits"].items())
        parent = manifest.get("parent")
        rows.append(
            {"version": version, "name": manifest["name"], "splits": splits, "parent": parent}
        )
    human = "\n".join(
        f"{r['version']}  {r['name']:<16} {r['splits']}"
        + (f"  (from {r['parent'][:15]})" if r["parent"] else "")
        for r in rows
    )
    emit(args, human or "no datasets yet", rows)
    return 0


# ---------------------------------------------------------------------------
# audit
# ---------------------------------------------------------------------------
@command("audit.verify")
def _audit_verify(args: argparse.Namespace, ws: Workspace) -> int:
    result = AuditLog(ws.audit_log).verify()
    data = {
        "ok": result.ok,
        "entries": result.entries,
        "head": result.head,
        "errors": result.errors,
    }
    if result.ok:
        human = f"audit log OK: {result.entries} entries, head {result.head}"
    else:
        human = "audit log BROKEN:\n" + "\n".join(f"  {e}" for e in result.errors)
    emit(args, human, data)
    return 0 if result.ok else 1


@command("audit.head")
def _audit_head(args: argparse.Namespace, ws: Workspace) -> int:
    print(AuditLog(ws.audit_log).head())
    return 0


@command("audit.log")
def _audit_log(args: argparse.Namespace, ws: Workspace) -> int:
    entries = list(AuditLog(ws.audit_log).entries())[-args.tail :]
    human = "\n".join(
        f"{e.seq:>5} {e.ts} {e.actor:<12} {e.event:<28} "
        + " ".join(f"{k}={v[:18]}" for k, v in e.subjects.items())
        for e in entries
    )
    emit(args, human or "audit log is empty", [e.to_dict() for e in entries])
    return 0


# ---------------------------------------------------------------------------
# parser
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    """The full argument parser."""
    parser = argparse.ArgumentParser(
        prog="lineage", description="Secured MLOps lifecycle with a hash-chained audit log."
    )
    parser.add_argument("--version", action="version", version=f"lineage {__version__}")
    parser.add_argument("-C", "--workspace", type=Path, help="project directory")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    groups = parser.add_subparsers(dest="group", required=True)

    data = groups.add_parser("data", help="datasets").add_subparsers(dest="cmd", required=True)
    p = data.add_parser("ingest", help="store raw files as an immutable dataset version")
    p.add_argument("--name", required=True)
    p.add_argument("splits", nargs="+", metavar="SPLIT=PATH")
    p = data.add_parser("validate", help="run the data checks, write report and data card")
    p.add_argument("version")
    p.add_argument("--fail-on-high", action="store_true", help="exit 1 on high findings")
    p = data.add_parser("card", help="print the data card")
    p.add_argument("version")
    p = data.add_parser("acknowledge", help="accept the open high findings, with a reason")
    p.add_argument("version")
    p.add_argument("--reason", required=True)
    p = data.add_parser("plant-canaries", help="derive a version with privacy canaries")
    p.add_argument("version")
    p.add_argument("--count", type=int, default=4)
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--seed", type=int)
    data.add_parser("list", help="list stored versions")

    audit = groups.add_parser("audit", help="the audit log").add_subparsers(
        dest="cmd", required=True
    )
    audit.add_parser("verify", help="recompute the whole hash chain")
    audit.add_parser("head", help="print the current head hash (publish it elsewhere)")
    p = audit.add_parser("log", help="show the last entries")
    p.add_argument("--tail", type=int, default=30)

    for extend in _EXTENSIONS:
        extend(groups)
    return parser


_EXTENSIONS: list[Callable[[Any], None]] = []


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point."""
    args = build_parser().parse_args(argv)
    ws = Workspace.load(args.workspace)
    handler = _COMMANDS[f"{args.group}.{args.cmd}"]
    try:
        return handler(args, ws)
    except PolicyDenied as exc:
        print(f"denied: {exc}", file=sys.stderr)
        for reason in exc.reasons:
            print(f"  - {reason}", file=sys.stderr)
        return 1
    except LineageError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
