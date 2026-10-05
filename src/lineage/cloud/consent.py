"""The consent gate: nothing leaves the machine without a named, scoped, expiring yes.

1. ``plan``     describes exactly what would leave: which dataset version, how many
                records, what personal data the validation found in it, where it goes
                (provider, region, bucket, encryption key) and what runs there. The plan
                is content-addressed: its id is the hash of what it says.
2. ``consent``  a named person agrees to *that plan id*, for a limited time, with a
                reason. If the plan carries personal data, they must say so explicitly.
                With ``approvers`` configured, only those people can consent, and never
                the person who drafted the plan.
3. ``apply``    re-checks the consent (exists, not expired, not revoked, plan file
                intact, dataset still validated) immediately before any network call.
                Dry run by default.

Consent is per plan: change one byte of what would be sent and it is a new plan that
needs a new consent. ``revoke`` withdraws a consent that was not used yet.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from lineage.audit import AuditLog
from lineage.data import service as data_service
from lineage.errors import IntegrityError, LineageError, PolicyDenied
from lineage.hashing import sha256_file, sha256_json
from lineage.supply import models
from lineage.workspace import Workspace, current_actor

ACTIONS = ("train",)


def _now() -> datetime:
    return datetime.now(UTC).replace(microsecond=0)


def settings(ws: Workspace) -> dict[str, Any]:
    """The ``[cloud]`` table, with required keys checked."""
    section = ws.section("cloud")
    missing = [k for k in ("provider", "region", "bucket", "kms_key_id") if k not in section]
    if missing:
        raise LineageError(f"[cloud] is missing {missing}")
    if section["provider"] != "aws":
        raise LineageError("only provider = 'aws' is implemented")
    return section


def plan(ws: Workspace, dataset_ref: str, action: str = "train") -> dict[str, Any]:
    """Describe what would leave the machine, store it, return it."""
    if action not in ACTIONS:
        raise LineageError(f"action must be one of {ACTIONS}")
    config = settings(ws)
    allowed = list(config.get("allowed_regions", [config["region"]]))
    if config["region"] not in allowed:
        raise PolicyDenied(f"region {config['region']} is not in allowed_regions {allowed}")
    store = data_service.store_of(ws)
    version = store.resolve(dataset_ref)
    report = data_service.require_trainable(ws, version)
    manifest = store.manifest(version)
    files = {
        name: {"sha256": digest, "bytes": (store.path(version) / name).stat().st_size}
        for name, digest in manifest["files"].items()
    }
    pii = {split: kinds for split, kinds in report["pii"].items() if kinds}
    pin, snapshot = models.verify_snapshot(ws)
    image = str(config.get("training_image", ""))
    if "@sha256:" not in image:
        raise PolicyDenied("[cloud].training_image must be pinned by digest (...@sha256:...)")
    if not config.get("role_arn"):
        raise LineageError("[cloud] needs role_arn for the training job")
    body: dict[str, Any] = {
        "action": action,
        "provider": "aws",
        "region": config["region"],
        "destination": {
            "bucket": config["bucket"],
            "prefix": f"{config.get('prefix', 'lineage').strip('/')}/{version}",
            "kms_key_id": config["kms_key_id"],
        },
        "data": {
            "dataset": version,
            "splits": manifest["splits"],
            "files": files,
            "personal_data": pii,
            "validation_report": report["report_hash"],
        },
        "base_model": {"ref": pin.ref, "slug": snapshot.name, "files": pin.files},
        "compute": {
            "service": "sagemaker",
            "instance_type": config.get("instance_type", "ml.m5.xlarge"),
            "max_runtime_seconds": int(config.get("max_runtime_seconds", 3600)),
            "network_isolation": True,
            "role_arn": config.get("role_arn"),
            "image": config.get("training_image"),
        },
        "drafted_by": current_actor(),
    }
    plan_id = "plan-" + sha256_json(body)[:24]
    stored = {"id": plan_id, **body, "drafted_at": _now().isoformat()}
    ws.consents.mkdir(parents=True, exist_ok=True)
    path = ws.consents / f"{plan_id}.json"
    path.write_text(json.dumps(stored, indent=2) + "\n")
    AuditLog(ws.audit_log).append(
        "cloud.planned",
        subjects={"plan": plan_id, "dataset": version},
        payload={
            "provider": "aws",
            "region": config["region"],
            "bucket": config["bucket"],
            "personal_data": pii,
            "plan_sha256": sha256_file(path),
        },
    )
    return stored


def load_plan(ws: Workspace, plan_id: str) -> dict[str, Any]:
    """A stored plan, re-checked against its id."""
    path = ws.consents / f"{plan_id}.json"
    if not path.exists():
        raise LineageError(f"no plan {plan_id}")
    stored: dict[str, Any] = json.loads(path.read_text())
    body = {k: v for k, v in stored.items() if k not in {"id", "drafted_at"}}
    if "plan-" + sha256_json(body)[:24] != plan_id:
        raise IntegrityError(f"{plan_id} was edited after it was drafted")
    return stored


def _events(ws: Workspace, plan_id: str) -> list[Any]:
    return [e for e in AuditLog(ws.audit_log).entries() if e.subjects.get("plan") == plan_id]


def grant(
    ws: Workspace,
    plan_id: str,
    reason: str,
    *,
    ttl_hours: float | None = None,
    acknowledge_personal_data: bool = False,
) -> dict[str, Any]:
    """Record a named consent to one plan."""
    stored = load_plan(ws, plan_id)
    config = settings(ws)
    actor = current_actor()
    approvers = list(config.get("approvers", []))
    if approvers and actor not in approvers:
        raise PolicyDenied(f"{actor} is not in [cloud].approvers {approvers}")
    if approvers and actor == stored["drafted_by"]:
        raise PolicyDenied("the person who drafted a plan cannot consent to it")
    if len(reason.strip()) < 10:
        raise LineageError("consent needs a reason (10+ characters)")
    if stored["data"]["personal_data"] and not acknowledge_personal_data:
        raise PolicyDenied(
            "this plan sends personal data "
            f"({json.dumps(stored['data']['personal_data'])}); consent must acknowledge it "
            "explicitly (--acknowledge-personal-data)",
        )
    ttl = float(ttl_hours if ttl_hours is not None else config.get("consent_ttl_hours", 24))
    if ttl <= 0 or ttl > float(config.get("max_consent_ttl_hours", 72)):
        raise PolicyDenied(
            f"consent lifetime must be between 0 and "
            f"{config.get('max_consent_ttl_hours', 72)} hours"
        )
    expires = _now() + timedelta(hours=ttl)
    record = {
        "plan": plan_id,
        "by": actor,
        "reason": reason.strip(),
        "expires": expires.isoformat(),
        "acknowledged_personal_data": acknowledge_personal_data,
    }
    AuditLog(ws.audit_log).append("cloud.consented", subjects={"plan": plan_id}, payload=record)
    return record


def revoke(ws: Workspace, plan_id: str, reason: str) -> None:
    """Withdraw consent to a plan."""
    load_plan(ws, plan_id)
    AuditLog(ws.audit_log).append(
        "cloud.revoked", subjects={"plan": plan_id}, payload={"reason": reason}
    )


def check(ws: Workspace, plan_id: str, now: datetime | None = None) -> dict[str, Any]:
    """Raise unless the plan has a valid, unused, unexpired, unrevoked consent."""
    stored = load_plan(ws, plan_id)
    now = now or _now()
    consent = None
    for entry in _events(ws, plan_id):
        if entry.event == "cloud.consented":
            consent = entry.payload
        elif entry.event == "cloud.revoked":
            consent = None
        elif entry.event == "cloud.executed":
            consent = None  # a consent is good for one execution
    if consent is None:
        raise PolicyDenied(f"{plan_id} has no valid consent (never given, revoked or used)")
    if datetime.fromisoformat(consent["expires"]) <= now:
        raise PolicyDenied(f"consent to {plan_id} expired at {consent['expires']}")
    # The data must still be what the plan describes, and still pass the training gate.
    report = data_service.require_trainable(ws, stored["data"]["dataset"])
    if report["report_hash"] != stored["data"]["validation_report"]:
        raise PolicyDenied(
            f"{stored['data']['dataset']} was re-validated with a different "
            "result since the plan was drafted; draft a new plan"
        )
    store = data_service.store_of(ws)
    for name, info in stored["data"]["files"].items():
        path = store.path(stored["data"]["dataset"]) / name
        if sha256_file(path) != info["sha256"]:
            raise IntegrityError(f"{name} changed since the plan was drafted")
    pin, _ = models.verify_snapshot(ws)
    if pin.ref != stored["base_model"]["ref"]:
        raise PolicyDenied(
            f"the workspace now pins {pin.ref}; the plan was for {stored['base_model']['ref']}"
        )
    return {"plan": stored, "consent": consent}


def plan_path(ws: Workspace, plan_id: str) -> Path:
    """Where a plan is stored."""
    return ws.consents / f"{plan_id}.json"


def import_run(ws: Workspace, directory: Path, plan_id: str) -> str:
    """Bring a cloud training run back, verifying it is what the plan said it would be.

    ``directory`` is the job's output (``run.json`` + ``adapter/``), downloaded from the
    plan's S3 output prefix. The adapter must match the hashes in ``run.json``, and the
    run must name the plan's dataset version and base model. It then becomes an
    ordinary local run: evaluation, registry and promotion apply unchanged.
    """
    import shutil

    stored = load_plan(ws, plan_id)
    executed = [e for e in _events(ws, plan_id) if e.event == "cloud.executed"]
    if not executed:
        raise PolicyDenied(f"{plan_id} was never executed; there is no cloud run to import")
    record = json.loads((directory / "run.json").read_text())
    for name, digest in record["adapter_files"].items():
        if sha256_file(directory / "adapter" / name) != digest:
            raise IntegrityError(f"adapter/{name} does not match the run record")
    if record["dataset"] != stored["data"]["dataset"]:
        raise PolicyDenied(
            f"the run trained on {record['dataset']}, the plan sent {stored['data']['dataset']}"
        )
    if record["base_model"] != stored["base_model"]["ref"]:
        raise PolicyDenied(
            f"the run used {record['base_model']}, the plan sent {stored['base_model']['ref']}"
        )
    cloud_log = directory / "cloud-audit.jsonl"
    if cloud_log.exists():
        chain = AuditLog(cloud_log).verify()
        if not chain.ok:
            raise IntegrityError(f"the job's own audit log is broken: {chain.errors[0]}")
    target = ws.runs / record["run_id"]
    if target.exists():
        raise LineageError(f"{record['run_id']} already exists locally")
    shutil.copytree(directory, target)
    log = AuditLog(ws.audit_log)
    payload = {
        "source": "aws",
        "plan": plan_id,
        "config_hash": record["config_hash"],
        "adapter_files": record["adapter_files"],
        "metrics": record["metrics"],
        "run_record_sha256": sha256_file(target / "run.json"),
    }
    log.append(
        "cloud.run_imported",
        subjects={
            "plan": plan_id,
            "run": record["run_id"],
            "dataset": record["dataset"],
            "base_model": record["base_model"],
        },
        payload=payload,
    )
    log.append(
        "train.finished",
        subjects={
            "run": record["run_id"],
            "dataset": record["dataset"],
            "base_model": record["base_model"],
        },
        payload=payload,
    )
    return str(record["run_id"])
