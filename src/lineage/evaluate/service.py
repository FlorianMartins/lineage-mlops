"""Evaluate a training run and decide its gates.

The report is bound to exactly what was evaluated (adapter digest, dataset version,
base model, red-team suite hash, thresholds), hashed, stored next to the run, and
logged. The registry later refuses to promote a model whose report does not match its
adapter, did not pass, or was edited.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from lineage.audit import AuditLog
from lineage.data import service as data_service
from lineage.data.store import Record
from lineage.errors import IntegrityError, LineageError, PolicyDenied
from lineage.evaluate import gates, privacy, safety
from lineage.evaluate.predictor import Model, Prompter, first_answer
from lineage.hashing import sha256_file, sha256_json
from lineage.supply import models
from lineage.task import TaskSpec
from lineage.train import service as train_service
from lineage.workspace import Workspace

REPORT = "eval.json"
# Untuned systems answer and then start the next few-shot example: stop there. Their
# outputs are cut at "###" anyway (first_answer), so this only saves time.
STOP = ["###"]


def quality_metrics(records: list[Record], outputs: list[str], task: TaskSpec) -> dict[str, Any]:
    """Exact match (every field right), per-field accuracy and well-formedness."""
    n = len(records)
    per_field = dict.fromkeys(task.fields, 0)
    exact = valid = 0
    for record, output in zip(records, outputs, strict=True):
        truth = task.parse(record.output)
        guess = task.parse(output)
        hits = [guess.get(k) == v for k, v in truth.items()]
        exact += all(hits)
        valid += not task.problems(output)
        for name, hit in zip(truth, hits, strict=True):
            per_field[name] += hit
    return {
        "examples": n,
        "exact_match": round(exact / n, 4) if n else 0.0,
        "valid": round(valid / n, 4) if n else 0.0,
        "field_accuracy": {k: round(v / n, 4) if n else 0.0 for k, v in per_field.items()},
    }


def reference_score(ws: Workspace) -> tuple[str | None, float | None]:
    """Exact match of the current production model, for the regression check."""
    try:
        from lineage.registry.store import Registry
    except ImportError:  # pragma: no cover - registry is part of the package
        return None, None
    registry = Registry(ws)
    current = registry.production()
    if current is None:
        return None, None
    report = registry.eval_report(current)
    return current["version"], float(report["results"]["quality"]["finetuned"]["exact_match"])


def evaluate(ws: Workspace, run_ref: str) -> dict[str, Any]:
    """Run all three gates on a run; write and log the report."""
    run = train_service.load_run(ws, run_ref)
    pin, snapshot = models.verify_snapshot(ws)
    if run.record["base_model"] != pin.ref:
        raise PolicyDenied(
            f"{run.id} was trained on {run.record['base_model']}, the workspace now pins {pin.ref}"
        )
    task = data_service.task_of(ws)
    settings = ws.section("eval")
    config = ws.section("gates")
    dataset = data_service.store_of(ws).get(run.record["dataset"])
    heldout_name = str(settings.get("heldout_split", "heldout"))
    if heldout_name not in dataset.split_names:
        raise LineageError(f"{dataset.version} has no '{heldout_name}' split to evaluate on")
    heldout = list(dataset.split(heldout_name).records)
    train_records = list(dataset.split("train").records)
    suite_path = ws.resolve(str(settings.get("redteam", "redteam.jsonl")))
    suite = safety.load_suite(suite_path)
    canaries = data_service.load_canaries(ws, dataset.version)
    threads = int(ws.section("train").get("threads", 4))

    started = time.monotonic()
    base = Model.load(snapshot, None, "base", threads)
    finetuned = Model.load(snapshot, run.adapter, "finetuned", threads)
    prompter = Prompter(
        task,
        base.tokenizer,
        [(r.input, r.output) for r in train_records],
        k=int(settings.get("rag_k", 4)),
        baseline_format=str(settings.get("baseline_format", "raw")),
    )

    # Quality: three systems on the same held-out questions.
    outputs = {
        "finetuned": finetuned.generate([prompter.finetuned(r.input) for r in heldout]),
        "base": [
            first_answer(o)
            for o in base.generate([prompter.base(r.input) for r in heldout], stop=STOP)
        ],
        "rag": [
            first_answer(o)
            for o in base.generate([prompter.rag(r.input) for r in heldout], stop=STOP)
        ],
    }
    quality_results = {k: quality_metrics(heldout, v, task) for k, v in outputs.items()}

    # Privacy: canaries and PII probes, against the base model as the chance level.
    both = {"finetuned": finetuned, "base": base}
    canary_results = (
        privacy.canary_report(
            both, canaries, task, candidates=int(settings.get("canary_candidates", 255))
        )
        if canaries
        else None
    )
    pii_results = privacy.pii_report(
        both, train_records, task, max_probes=int(settings.get("pii_probes", 40))
    )

    # Safety: the red-team suite, each system prompted as it is used.
    redteam = {
        "finetuned": safety.score(
            suite,
            finetuned.generate([prompter.finetuned(c["input"]) for c in suite], max_new_tokens=48),
            task,
        ),
        "base": safety.score(
            suite,
            [
                first_answer(o)
                for o in base.generate(
                    [prompter.base(c["input"]) for c in suite], max_new_tokens=48, stop=STOP
                )
            ],
            task,
        ),
    }

    production_version, production_score = reference_score(ws)
    decided = {
        "quality": gates.quality(quality_results, config, production_score),
        "privacy": gates.privacy(canary_results, pii_results, config),
        "safety": gates.safety(redteam, config),
    }
    report: dict[str, Any] = {
        "run": run.id,
        "adapter_digest": train_service.adapter_digest(run),
        "dataset": dataset.version,
        "base_model": pin.ref,
        "redteam_suite": {
            "path": str(suite_path.relative_to(ws.root))
            if suite_path.is_relative_to(ws.root)
            else str(suite_path),
            "sha256": sha256_file(suite_path),
            "cases": len(suite),
        },
        "reference": {"production": production_version, "exact_match": production_score},
        "results": {
            "quality": quality_results,
            "canaries": canary_results,
            "pii": pii_results,
            "redteam": redteam,
        },
        "gates": {name: gate.to_dict() for name, gate in decided.items()},
        "passed": all(g.passed for g in decided.values()),
        "seconds": round(time.monotonic() - started, 1),
    }
    report["report_hash"] = sha256_json(report)
    (run.path / REPORT).write_text(json.dumps(report, indent=2) + "\n")
    AuditLog(ws.audit_log).append(
        "eval.completed",
        subjects={"run": run.id, "dataset": dataset.version, "base_model": pin.ref},
        payload={
            "report_hash": report["report_hash"],
            "adapter_digest": report["adapter_digest"],
            "passed": report["passed"],
            "gates": {k: {"passed": g.passed, "failures": g.failures} for k, g in decided.items()},
            "exact_match": {k: v["exact_match"] for k, v in quality_results.items()},
        },
    )
    return report


def load_report(run_path: Path, adapter_digest: str) -> dict[str, Any]:
    """Load a report and check it is intact and about this exact adapter."""
    path = run_path / REPORT
    if not path.exists():
        raise PolicyDenied(f"{run_path.name} has not been evaluated (run `lineage eval run`)")
    report: dict[str, Any] = json.loads(path.read_text())
    body = {k: v for k, v in report.items() if k != "report_hash"}
    if sha256_json(body) != report.get("report_hash"):
        raise IntegrityError(f"{run_path.name}/{REPORT} was edited after evaluation")
    if report["adapter_digest"] != adapter_digest:
        raise IntegrityError(f"{run_path.name}: the evaluation is about a different adapter")
    return report
