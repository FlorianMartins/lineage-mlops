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
# model (supply chain)
# ---------------------------------------------------------------------------
@command("model.pin")
def _model_pin(args: argparse.Namespace, ws: Workspace) -> int:
    import tempfile

    from lineage.supply.models import HuggingFaceHub, pin_block

    with tempfile.TemporaryDirectory() as tmp:
        text, skipped = pin_block(HuggingFaceHub(), args.repo, args.revision, Path(tmp))
    print(text)
    for item in skipped:
        print(f"# left out: {item}")
    return 0


@command("model.fetch")
def _model_fetch(args: argparse.Namespace, ws: Workspace) -> int:
    from lineage.supply.models import HuggingFaceHub, fetch

    path = fetch(ws, HuggingFaceHub(), use=args.use)
    emit(args, f"verified snapshot at {path}", {"path": str(path)})
    return 0


@command("model.verify")
def _model_verify(args: argparse.Namespace, ws: Workspace) -> int:
    from lineage.supply.models import verify_snapshot

    pin, path = verify_snapshot(ws)
    emit(args, f"{pin.ref}: every file matches its pin ({path})", {"ref": pin.ref})
    return 0


@command("model.accept-licence")
def _model_accept(args: argparse.Namespace, ws: Workspace) -> int:
    from lineage.supply.models import accept_licence

    accept_licence(ws, args.reason)
    print("licence conditions accepted and recorded")
    return 0


@command("model.scan")
def _model_scan(args: argparse.Namespace, ws: Workspace) -> int:
    from lineage.supply import weights

    target = Path(args.path)
    data: dict[str, Any]
    if target.is_dir():
        inspection = weights.inspect(target)
        scans = [weights.scan_pickle(target / p).to_dict() for p in inspection.pickles]
        data = {
            "ok": inspection.ok,
            "safetensors": inspection.safetensors,
            "pickles": scans,
            "unknown": inspection.unknown,
            "problems": inspection.problems,
        }
    else:
        if target.suffix == ".safetensors":
            problems = weights.check_safetensors(target)
            data = {"ok": not problems, "problems": problems}
        else:
            scan = weights.scan_pickle(target)
            data = {
                "ok": False,
                "pickles": [scan.to_dict()],
                "problems": ["pickle-based file: never loaded by Lineage"],
            }
    lines = [f"{args.path}: {'loadable' if data['ok'] else 'REFUSED'}"]
    for item in data.get("pickles", []):
        lines.append(f"  pickle {item['path']} imports {', '.join(item['globals']) or 'nothing'}")
        if item["dangerous"]:
            lines.append(f"    DANGEROUS: {', '.join(item['dangerous'])}")
    lines += [f"  {p}" for p in data.get("problems", [])]
    emit(args, "\n".join(lines), data)
    return 0 if data["ok"] else 1


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------
@command("train.run")
def _train_run(args: argparse.Namespace, ws: Workspace) -> int:
    from lineage.train import service

    run = service.train(ws, args.dataset, epochs=args.epochs, seed=args.seed, loss_on=args.loss_on)
    metrics = run.record["metrics"]
    emit(
        args,
        f"{run.id}  loss {metrics['first_loss']} -> {metrics['final_loss']} "
        f"in {run.record['seconds']}s ({metrics['steps']} steps)\n"
        f"  config hash {run.record['config_hash']}\n"
        f"  adapter     {run.adapter}",
        run.record,
    )
    return 0


@command("train.list")
def _train_list(args: argparse.Namespace, ws: Workspace) -> int:
    from lineage.train import service

    rows = service.runs(ws)
    human = "\n".join(
        f"{r['run_id']}  {r['dataset'][:15]}  loss {r['metrics']['final_loss']}  "
        f"config {r['config_hash'][:12]}"
        for r in rows
    )
    emit(args, human or "no runs yet", rows)
    return 0


# ---------------------------------------------------------------------------
# eval
# ---------------------------------------------------------------------------
def _gate_lines(report: dict[str, Any]) -> list[str]:
    quality = report["results"]["quality"]
    lines = [
        f"{report['run']}  {'PASSED' if report['passed'] else 'FAILED'}  "
        f"({report['seconds']}s, report {report['report_hash'][:12]})",
        "  quality   exact match  finetuned {:.3f} | base {:.3f} | rag {:.3f}".format(
            quality["finetuned"]["exact_match"],
            quality["base"]["exact_match"],
            quality["rag"]["exact_match"],
        ),
    ]
    canaries = report["results"]["canaries"]
    if canaries:
        lines.append(
            "  privacy   canary exposure max {:.2f} bits (base {:.2f}), extracted {}".format(
                canaries["finetuned"]["max_exposure"],
                canaries["base"]["max_exposure"],
                canaries["finetuned"]["extracted"],
            )
        )
    pii_result = report["results"]["pii"]
    lines.append(
        f"            PII leak rate {pii_result['finetuned']['leak_rate']:.3f} "
        f"(base {pii_result['base']['leak_rate']:.3f}, {pii_result['probes']} probes)"
    )
    redteam = report["results"]["redteam"]
    lines.append(
        "  safety    attack success {:.3f} (base {:.3f}) {}".format(
            redteam["finetuned"]["attack_success_rate"],
            redteam["base"]["attack_success_rate"],
            redteam["finetuned"]["by_kind"],
        )
    )
    for name, gate in report["gates"].items():
        mark = "pass" if gate["passed"] else "FAIL"
        lines.append(f"  gate {name:<8} {mark}")
        lines += [f"      - {f}" for f in gate["failures"]]
    return lines


@command("eval.run")
def _eval_run(args: argparse.Namespace, ws: Workspace) -> int:
    from lineage.evaluate import service

    report = service.evaluate(ws, args.run)
    emit(args, "\n".join(_gate_lines(report)), report)
    return 0 if report["passed"] else 1


@command("eval.show")
def _eval_show(args: argparse.Namespace, ws: Workspace) -> int:
    from lineage.evaluate import service
    from lineage.train import service as train_service

    run = train_service.load_run(ws, args.run)
    report = service.load_report(run.path, train_service.adapter_digest(run))
    emit(args, "\n".join(_gate_lines(report)), report)
    return 0


# ---------------------------------------------------------------------------
# registry and signing
# ---------------------------------------------------------------------------
@command("signing.init")
def _signing_init(args: argparse.Namespace, ws: Workspace) -> int:
    from lineage.registry.signing import init_keys

    path = init_keys(ws)
    print(f"cosign key pair written to {path} (private key mode 0600)")
    return 0


@command("registry.register")
def _registry_register(args: argparse.Namespace, ws: Workspace) -> int:
    from lineage.registry import service
    from lineage.registry.store import Registry

    number = service.register(ws, args.run)
    name = Registry(ws).model_name
    emit(
        args,
        f"{name}:{number} registered as candidate (signed, ML-BOM written)",
        {"model": name, "version": number},
    )
    return 0


@command("registry.approve")
def _registry_approve(args: argparse.Namespace, ws: Workspace) -> int:
    from lineage.registry import service

    service.approve(ws, args.ref, args.reason)
    print(f"approval of {args.ref} recorded")
    return 0


@command("registry.promote")
def _registry_promote(args: argparse.Namespace, ws: Workspace) -> int:
    from lineage.registry import service

    result = service.promote(ws, args.ref, args.to)
    replaced = f" (replaces version {result['replaced']})" if result["replaced"] else ""
    emit(
        args,
        f"{result['model']}:{result['version']} {result['from']} -> {result['to']}{replaced}",
        result,
    )
    return 0


@command("registry.rollback")
def _registry_rollback(args: argparse.Namespace, ws: Workspace) -> int:
    from lineage.registry import service

    result = service.rollback(ws, args.model, args.reason)
    emit(
        args,
        f"{result['model']}: production is version {result['version']} again "
        f"(version {result['replaced']} rolled back)",
        result,
    )
    return 0


@command("registry.list")
def _registry_list(args: argparse.Namespace, ws: Workspace) -> int:
    from lineage.registry.store import Registry

    registry = Registry(ws)
    rows = registry.versions(args.model)
    human = "\n".join(
        f"{registry.model_name if not args.model else args.model}:{r['version']:<3} "
        f"{r['stage']:<12} run {r['run']}  approvals {len(r['approvals'])}"
        for r in rows
    )
    emit(args, human or "nothing registered yet", rows)
    return 0


@command("registry.verify")
def _registry_verify(args: argparse.Namespace, ws: Workspace) -> int:
    from lineage.registry.store import Registry

    registry = Registry(ws)
    name, number = registry.resolve(args.ref)
    result = registry.verify(number, name)
    lines = [f"{name}:{number}: {'verified' if result['ok'] else 'NOT VERIFIED'}"]
    for key in ("manifest_intact", "signature_verified", "provenance_verified"):
        lines.append(f"  {key:<20} {result[key]}")
    lines += [f"  - {p}" for p in result["problems"]]
    emit(args, "\n".join(lines), result)
    return 0 if result["ok"] else 1


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

    model = groups.add_parser("model", help="base model supply chain").add_subparsers(
        dest="cmd", required=True
    )
    p = model.add_parser("pin", help="print a [base_model] block with file hashes")
    p.add_argument("repo")
    p.add_argument("--revision", required=True, help="exact 40-character commit SHA")
    p = model.add_parser("fetch", help="download, verify and admit the pinned model")
    p.add_argument("--use", choices=["research", "internal", "commercial", "redistribution"])
    model.add_parser("verify", help="re-hash the local snapshot against the pin")
    p = model.add_parser("accept-licence", help="record acceptance of licence conditions")
    p.add_argument("--reason", required=True)
    p = model.add_parser("scan", help="inspect weights or a pickle without loading it")
    p.add_argument("path")

    train = groups.add_parser("train", help="training runs").add_subparsers(
        dest="cmd", required=True
    )
    p = train.add_parser("run", help="train a LoRA adapter on a validated dataset")
    p.add_argument("dataset")
    p.add_argument("--epochs", type=int)
    p.add_argument("--seed", type=int)
    p.add_argument("--loss-on", choices=["completion", "full"])
    train.add_parser("list", help="list runs")

    evaluation = groups.add_parser("eval", help="evaluation gates").add_subparsers(
        dest="cmd", required=True
    )
    p = evaluation.add_parser("run", help="quality, privacy and safety gates for a run")
    p.add_argument("run")
    p = evaluation.add_parser("show", help="show a stored evaluation report")
    p.add_argument("run")

    signing = groups.add_parser("signing", help="signing keys").add_subparsers(
        dest="cmd", required=True
    )
    signing.add_parser("init", help="generate a cosign key pair (key mode)")

    registry = groups.add_parser("registry", help="model registry").add_subparsers(
        dest="cmd", required=True
    )
    p = registry.add_parser("register", help="register an evaluated run as a candidate")
    p.add_argument("run")
    p = registry.add_parser("approve", help="approve a version's manifest")
    p.add_argument("ref", help="name:version")
    p.add_argument("--reason", required=True)
    p = registry.add_parser("promote", help="promote a version, if the policy allows")
    p.add_argument("ref", help="name:version")
    p.add_argument("--to", required=True, choices=["staging", "production"])
    p = registry.add_parser("rollback", help="restore the previous production version")
    p.add_argument("model", nargs="?")
    p.add_argument("--reason", required=True)
    p = registry.add_parser("list", help="versions and stages")
    p.add_argument("model", nargs="?")
    p = registry.add_parser("verify", help="re-hash files and verify signatures")
    p.add_argument("ref", help="name:version or name (production)")

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
