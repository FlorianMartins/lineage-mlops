"""A training run, end to end: gates, training, tracking, evidence.

Before a single step runs, the run checks that the dataset was validated (and any high
finding acknowledged) and that the base model snapshot still matches its pin. After
training, the adapter's hashes, the config hash, the environment and the metrics go to
three places that must agree: ``run.json`` next to the adapter, MLflow, and the audit
log.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from lineage.audit import AuditLog
from lineage.data import service as data_service
from lineage.errors import IntegrityError, LineageError
from lineage.hashing import sha256_file, sha256_json
from lineage.supply import models
from lineage.train import env
from lineage.train.config import TrainConfig, config_hash
from lineage.workspace import Workspace


@dataclass(frozen=True)
class Run:
    """A finished run as recorded in ``run.json``."""

    id: str
    path: Path
    record: dict[str, Any]

    @property
    def adapter(self) -> Path:
        """Directory holding the LoRA adapter."""
        return self.path / "adapter"


def _tracking(ws: Workspace) -> tuple[str, str]:
    tracking = ws.section("tracking")
    uri = str(tracking.get("uri", f"sqlite:///{ws.state / 'mlflow.db'}"))
    experiment = str(tracking.get("experiment", ws.section("task").get("name", "lineage")))
    return uri, experiment


def _lock_file(ws: Workspace) -> Path | None:
    value = ws.section("train").get("lock_file")
    return ws.resolve(str(value)) if value else None


def train(ws: Workspace, dataset_ref: str, **overrides: Any) -> Run:
    """Run the gates, train, record. Returns the run."""
    store = data_service.store_of(ws)
    dataset_version = store.resolve(dataset_ref)
    report = data_service.require_trainable(ws, dataset_version)
    pin, snapshot = models.verify_snapshot(ws)
    task = data_service.task_of(ws)
    config = TrainConfig.from_config(ws.section("train"), **overrides)
    lock = _lock_file(ws)
    environment = env.capture(ws.root, lock)
    digest = config_hash(
        config,
        dataset=dataset_version,
        base_model=pin.ref,
        task={
            "fields": {k: list(v) for k, v in task.fields.items()},
            "template": task.prompt_template,
        },
        lock_sha256=environment["lock_sha256"],
    )
    started = datetime.now(UTC)
    run_id = f"run-{started:%Y%m%dT%H%M%S}-{digest[:8]}"
    run_dir = ws.runs / run_id
    if run_dir.exists():
        raise LineageError(f"{run_id} already exists")
    dataset = store.get(dataset_version)
    examples = [(r.input, r.output) for r in dataset.split("train").records]

    log = AuditLog(ws.audit_log)
    log.append(
        "train.started",
        subjects={"run": run_id, "dataset": dataset_version, "base_model": pin.ref},
        payload={
            "config_hash": digest,
            "config": config.to_dict(),
            "validation_report": report["report_hash"],
            "code": environment["code"],
            "lock_sha256": environment["lock_sha256"],
        },
    )

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    import mlflow

    from lineage.train import lora

    uri, experiment = _tracking(ws)
    mlflow.set_tracking_uri(uri)
    if mlflow.get_experiment_by_name(experiment) is None:
        mlflow.create_experiment(experiment, artifact_location=(ws.state / "mlartifacts").as_uri())
    mlflow.set_experiment(experiment)
    t0 = time.monotonic()
    try:
        with mlflow.start_run(run_name=run_id) as active:
            mlflow.set_tags(
                {
                    "lineage.run_id": run_id,
                    "lineage.config_hash": digest,
                    "lineage.dataset": dataset_version,
                    "lineage.base_model": pin.ref,
                    "lineage.code_commit": str(environment["code"]["commit"]),
                    "lineage.code_dirty": str(environment["code"]["dirty"]),
                    "lineage.lock_sha256": str(environment["lock_sha256"]),
                }
            )
            mlflow.log_params({k: str(v) for k, v in config.to_dict().items()})

            def _log_loss(step: int, loss: float) -> None:
                mlflow.log_metric("train_loss", loss, step=step)

            result = lora.train(
                snapshot,
                examples,
                task,
                config,
                run_dir / "adapter",
                base_ref=pin.ref,
                on_step=_log_loss,
            )
            seconds = round(time.monotonic() - t0, 1)
            adapter_files = {
                p.name: sha256_file(p) for p in sorted((run_dir / "adapter").iterdir())
            }
            record = {
                "run_id": run_id,
                "config_hash": digest,
                "config": config.to_dict(),
                "dataset": dataset_version,
                "validation_report": report["report_hash"],
                "base_model": pin.ref,
                "base_model_files": pin.files,
                "environment": environment,
                "mlflow": {
                    "tracking_uri": uri,
                    "experiment": experiment,
                    "run_id": active.info.run_id,
                },
                "started": started.isoformat(),
                "seconds": seconds,
                "metrics": {
                    "steps": result.steps,
                    "final_loss": round(sum(result.losses[-5:]) / min(5, len(result.losses)), 4),
                    "first_loss": round(result.losses[0], 4),
                    "trainable_parameters": result.trainable_parameters,
                    "total_parameters": result.total_parameters,
                },
                "adapter_files": adapter_files,
            }
            (run_dir / "run.json").write_text(json.dumps(record, indent=2) + "\n")
            mlflow.log_metrics(
                {"final_loss": record["metrics"]["final_loss"], "train_seconds": seconds}
            )
            mlflow.log_artifact(str(run_dir / "run.json"))
            mlflow.log_artifacts(str(run_dir / "adapter"), artifact_path="adapter")
    except BaseException as exc:
        log.append(
            "train.failed",
            subjects={"run": run_id},
            payload={"error": f"{type(exc).__name__}: {exc}"[:500]},
        )
        shutil.rmtree(run_dir, ignore_errors=True)
        raise

    log.append(
        "train.finished",
        subjects={"run": run_id, "dataset": dataset_version, "base_model": pin.ref},
        payload={
            "config_hash": digest,
            "adapter_files": adapter_files,
            "metrics": record["metrics"],
            "mlflow_run_id": record["mlflow"]["run_id"],
            "run_record_sha256": sha256_file(run_dir / "run.json"),
        },
    )
    return Run(run_id, run_dir, record)


def load_run(ws: Workspace, ref: str) -> Run:
    """Load a run by id or unique prefix and re-verify its adapter files."""
    candidates = sorted(p for p in ws.runs.glob(f"{ref}*") if (p / "run.json").exists())
    if not candidates and not ref.startswith("run-"):
        candidates = sorted(p for p in ws.runs.glob(f"run-{ref}*") if (p / "run.json").exists())
    if len(candidates) != 1:
        raise LineageError(f"no run matches '{ref}'" if not candidates else f"'{ref}' is ambiguous")
    path = candidates[0]
    record = json.loads((path / "run.json").read_text())
    for name, digest in record["adapter_files"].items():
        if sha256_file(path / "adapter" / name) != digest:
            raise IntegrityError(f"{path.name}/adapter/{name} changed after training")
    return Run(path.name, path, record)


def runs(ws: Workspace) -> list[dict[str, Any]]:
    """Every run record, oldest first."""
    if not ws.runs.exists():
        return []
    return [
        json.loads((p / "run.json").read_text())
        for p in sorted(ws.runs.iterdir())
        if (p / "run.json").exists()
    ]


def adapter_digest(run: Run) -> str:
    """One hash for the whole adapter (used as the model's identity)."""
    return sha256_json(run.record["adapter_files"])
