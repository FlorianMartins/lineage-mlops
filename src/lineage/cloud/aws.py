"""AWS execution of a consented plan.

S3 upload (SSE-KMS, SHA-256 checksums) and a SageMaker training job with network
isolation.

Everything goes through :func:`calls`, which turns a plan into the exact list of API
calls. ``apply`` without ``execute=True`` returns that list and touches nothing (the
default); with it, the consent is re-checked and the calls are made with boto3.

What runs in the job is the image built from ``deploy/sagemaker/Dockerfile`` (this
repository + the locked ML stack), pinned by digest: a tag can be moved to other code,
a digest cannot. With network isolation the job cannot download anything, so the base
model travels as an input channel and is verified against the pin like any download.
The job trains only: evaluation needs the canary secrets, which never leave the
machine, so the result comes back through ``lineage cloud import-run`` and goes
through the local gates.

Exercised against botocore's Stubber in the test suite (request shapes, encryption and
isolation flags); not run against a real AWS account from this repository.
"""

from __future__ import annotations

import base64
import hashlib
from pathlib import Path
from typing import Any

from lineage.audit import AuditLog
from lineage.cloud import consent
from lineage.data import service as data_service
from lineage.workspace import Workspace


def _b64_sha256(path: Path) -> str:
    return base64.b64encode(hashlib.sha256(path.read_bytes()).digest()).decode()


def calls(ws: Workspace, stored: dict[str, Any]) -> list[dict[str, Any]]:
    """The API calls a plan turns into, in order."""
    dest = stored["destination"]
    store = data_service.store_of(ws)
    version = stored["data"]["dataset"]
    uploads: list[tuple[Path, str]] = [
        (store.path(version) / name, f"{dest['prefix']}/data/{name}")
        for name in stored["data"]["files"]
    ]
    snapshot = ws.models / stored["base_model"]["slug"]
    uploads += [
        (snapshot / name, f"{dest['prefix']}/model/{name}")
        for name in stored["base_model"]["files"]
    ]
    out: list[dict[str, Any]] = [
        {
            "service": "s3",
            "operation": "put_object",
            "params": {
                "Bucket": dest["bucket"],
                "Key": key,
                "ServerSideEncryption": "aws:kms",
                "SSEKMSKeyId": dest["kms_key_id"],
                "ChecksumAlgorithm": "SHA256",
                "ChecksumSHA256": _b64_sha256(path),
                "Tagging": f"lineage-plan={stored['id']}",
            },
            "body": str(path),
        }
        for path, key in uploads
    ]
    compute = stored["compute"]
    s3_root = f"s3://{dest['bucket']}/{dest['prefix']}"

    def channel(name: str) -> dict[str, Any]:
        return {
            "ChannelName": name,
            "DataSource": {
                "S3DataSource": {
                    "S3DataType": "S3Prefix",
                    "S3Uri": f"{s3_root}/{name}/",
                    "S3DataDistributionType": "FullyReplicated",
                }
            },
        }

    out.append(
        {
            "service": "sagemaker",
            "operation": "create_training_job",
            "params": {
                "TrainingJobName": f"lineage-{stored['id'][5:21]}",
                "RoleArn": compute["role_arn"],
                "AlgorithmSpecification": {
                    "TrainingImage": compute["image"],
                    "TrainingInputMode": "File",
                },
                "HyperParameters": {"lineage_dataset": version, "lineage_plan": stored["id"]},
                "InputDataConfig": [channel("data"), channel("model")],
                "OutputDataConfig": {
                    "S3OutputPath": f"{s3_root}/output/",
                    "KmsKeyId": dest["kms_key_id"],
                },
                "ResourceConfig": {
                    "InstanceType": compute["instance_type"],
                    "InstanceCount": 1,
                    "VolumeSizeInGB": 30,
                    "VolumeKmsKeyId": dest["kms_key_id"],
                },
                "StoppingCondition": {"MaxRuntimeInSeconds": compute["max_runtime_seconds"]},
                "EnableNetworkIsolation": True,
                "EnableInterContainerTrafficEncryption": True,
                "Tags": [
                    {"Key": "lineage-plan", "Value": stored["id"]},
                    {"Key": "lineage-dataset", "Value": version[:64]},
                ],
            },
        }
    )
    return out


def apply(
    ws: Workspace, plan_id: str, *, execute: bool = False, session: Any = None
) -> dict[str, Any]:
    """Dry run (default) or execute a consented plan."""
    checked = consent.check(ws, plan_id)
    stored = checked["plan"]
    planned = calls(ws, stored)
    if not execute:
        return {"plan": plan_id, "dry_run": True, "calls": planned}
    if session is None:
        import boto3

        session = boto3.session.Session(region_name=stored["region"])
    clients = {
        name: session.client(name, region_name=stored["region"]) for name in ("s3", "sagemaker")
    }
    # Re-check right before the first byte leaves (building the session can take a
    # while, and a revocation must win).
    consent.check(ws, plan_id)
    results = []
    for call in planned:
        params = dict(call["params"])
        if "body" in call:
            with Path(call["body"]).open("rb") as handle:
                response = getattr(clients[call["service"]], call["operation"])(
                    Body=handle, **params
                )
        else:
            response = getattr(clients[call["service"]], call["operation"])(**params)
        results.append(
            {
                "operation": call["operation"],
                "key": params.get("Key") or params.get("TrainingJobName"),
                "etag": response.get("ETag"),
                "arn": response.get("TrainingJobArn"),
            }
        )
    AuditLog(ws.audit_log).append(
        "cloud.executed",
        subjects={"plan": plan_id, "dataset": stored["data"]["dataset"]},
        payload={"results": results, "consent_by": checked["consent"]["by"]},
    )
    return {"plan": plan_id, "dry_run": False, "results": results}
