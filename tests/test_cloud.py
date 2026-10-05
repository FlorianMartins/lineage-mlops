import json
import shutil
from datetime import datetime, timedelta

import pytest

from lineage.audit import AuditLog
from lineage.cloud import aws, consent
from lineage.data import service as data_service
from lineage.errors import IntegrityError, LineageError, PolicyDenied
from lineage.workspace import Workspace

pytestmark = pytest.mark.ml

CLOUD = """[cloud]
provider = "aws"
region = "eu-west-3"
allowed_regions = ["eu-west-3"]
bucket = "test-bucket"
prefix = "lineage"
kms_key_id = "arn:aws:kms:eu-west-3:111122223333:key/abc"
role_arn = "arn:aws:iam::111122223333:role/Train"
training_image = "111122223333.dkr.ecr.eu-west-3.amazonaws.com/lineage@sha256:{digest}"
approvers = ["dpo", "security-lead"]
consent_ttl_hours = 24
"""


@pytest.fixture
def cloud_ws(tiny_workspace, clean_version, monkeypatch):
    from tests.conftest import replace_tables

    text = replace_tables(
        (tiny_workspace.root / "lineage.toml").read_text(), "cloud", CLOUD.format(digest="1" * 64)
    )
    (tiny_workspace.root / "lineage.toml").write_text(text)
    ws = Workspace.load(tiny_workspace.root)
    data_service.run_validation(ws, clean_version)
    monkeypatch.setenv("LINEAGE_ACTOR", "ml-engineer")
    return ws, clean_version


def as_actor(monkeypatch, name):
    monkeypatch.setenv("LINEAGE_ACTOR", name)


def test_plan_describes_what_leaves(cloud_ws):
    ws, version = cloud_ws
    stored = consent.plan(ws, version)
    assert stored["id"].startswith("plan-") and stored["data"]["dataset"] == version
    assert stored["data"]["personal_data"]["train"]["email"] > 0
    assert stored["base_model"]["ref"].startswith("test/tiny-llama@")
    assert consent.load_plan(ws, stored["id"]) == stored
    assert consent.plan(ws, version)["id"] == stored["id"]  # content-addressed


def test_plan_refuses_unpinned_images_and_other_regions(cloud_ws):
    ws, version = cloud_ws
    path = ws.root / "lineage.toml"
    text = path.read_text()
    path.write_text(text.replace(f"@sha256:{'1' * 64}", ":latest"))
    with pytest.raises(PolicyDenied, match="digest"):
        consent.plan(Workspace.load(ws.root), version)
    path.write_text(text.replace('region = "eu-west-3"', 'region = "us-east-1"'))
    with pytest.raises(PolicyDenied, match="allowed_regions"):
        consent.plan(Workspace.load(ws.root), version)


def test_consent_rules(cloud_ws, monkeypatch):
    ws, version = cloud_ws
    plan_id = consent.plan(ws, version)["id"]
    with pytest.raises(PolicyDenied, match="no valid consent"):
        consent.check(ws, plan_id)
    with pytest.raises(PolicyDenied, match="approvers"):
        consent.grant(ws, plan_id, "I drafted it and I like it")  # not an approver
    as_actor(monkeypatch, "dpo")
    with pytest.raises(PolicyDenied, match="personal data"):
        consent.grant(ws, plan_id, "training in our own AWS account")
    with pytest.raises(PolicyDenied, match="lifetime"):
        consent.grant(
            ws,
            plan_id,
            "training in our own AWS account",
            ttl_hours=1000,
            acknowledge_personal_data=True,
        )
    record = consent.grant(
        ws, plan_id, "training in our own AWS account, DPIA ref 12", acknowledge_personal_data=True
    )
    assert consent.check(ws, plan_id)["consent"]["by"] == "dpo"
    later = datetime.fromisoformat(record["expires"]) + timedelta(seconds=1)
    with pytest.raises(PolicyDenied, match="expired"):
        consent.check(ws, plan_id, now=later)
    consent.revoke(ws, plan_id, "project paused")
    with pytest.raises(PolicyDenied, match="no valid consent"):
        consent.check(ws, plan_id)


def test_drafter_cannot_consent_to_own_plan(cloud_ws, monkeypatch):
    ws, version = cloud_ws
    as_actor(monkeypatch, "dpo")
    plan_id = consent.plan(ws, version)["id"]
    with pytest.raises(PolicyDenied, match="drafted"):
        consent.grant(ws, plan_id, "my own plan, trust me", acknowledge_personal_data=True)


def test_edited_plan_and_changed_data_are_refused(cloud_ws, monkeypatch):
    ws, version = cloud_ws
    stored = consent.plan(ws, version)
    as_actor(monkeypatch, "dpo")
    consent.grant(
        ws, stored["id"], "training in our own AWS account", acknowledge_personal_data=True
    )
    path = consent.plan_path(ws, stored["id"])
    edited = dict(stored, region="us-east-1")
    path.write_text(json.dumps(edited))
    with pytest.raises(IntegrityError, match="edited"):
        consent.check(ws, stored["id"])
    path.write_text(json.dumps(stored))
    consent.check(ws, stored["id"])
    data_file = data_service.store_of(ws).path(version) / "train.jsonl"
    data_file.write_text(data_file.read_text().replace("VPN", "vpn", 1))
    with pytest.raises(IntegrityError):
        consent.check(ws, stored["id"])


def _stubbed_session(calls):
    import boto3
    from botocore.stub import ANY, Stubber

    session = boto3.session.Session(
        region_name="eu-west-3", aws_access_key_id="x", aws_secret_access_key="y"
    )
    clients = {name: session.client(name, region_name="eu-west-3") for name in ("s3", "sagemaker")}
    stubbers = {name: Stubber(client) for name, client in clients.items()}
    for call in calls:
        params = dict(call["params"])
        if "body" in call:
            params["Body"] = ANY
            response = {"ETag": '"etag"'}
        else:
            response = {"TrainingJobArn": "arn:aws:sagemaker:eu-west-3:1:training-job/x"}
        stubbers[call["service"]].add_response(call["operation"], response, params)
    for stubber in stubbers.values():
        stubber.activate()

    class Session:
        def client(self, name, region_name=None):
            return clients[name]

    return Session(), stubbers


def test_apply_dry_run_then_execute_once(cloud_ws, monkeypatch):
    ws, version = cloud_ws
    plan_id = consent.plan(ws, version)["id"]
    as_actor(monkeypatch, "security-lead")
    consent.grant(ws, plan_id, "training in our own AWS account", acknowledge_personal_data=True)
    dry = aws.apply(ws, plan_id)
    assert dry["dry_run"]
    puts = [c for c in dry["calls"] if c["operation"] == "put_object"]
    assert all(c["params"]["ServerSideEncryption"] == "aws:kms" for c in puts)
    assert all(c["params"]["ChecksumAlgorithm"] == "SHA256" for c in puts)
    assert {c["params"]["Key"].split("/")[2] for c in puts} == {"data", "model"}
    job = dry["calls"][-1]["params"]
    assert job["EnableNetworkIsolation"] and job["EnableInterContainerTrafficEncryption"]
    assert job["HyperParameters"]["lineage_dataset"] == version
    assert not any(e.event == "cloud.executed" for e in AuditLog(ws.audit_log).entries())

    session, stubbers = _stubbed_session(dry["calls"])
    result = aws.apply(ws, plan_id, execute=True, session=session)
    for stubber in stubbers.values():
        stubber.assert_no_pending_responses()
    assert result["results"][-1]["arn"].startswith("arn:aws:sagemaker")
    with pytest.raises(PolicyDenied, match="no valid consent"):
        aws.apply(ws, plan_id)  # a consent is good for one execution


def test_import_run_verifies_and_rejoins_the_local_gates(cloud_ws, monkeypatch, tmp_path):
    from lineage.train import service as train_service

    ws, version = cloud_ws
    stored = consent.plan(ws, version)
    as_actor(monkeypatch, "dpo")
    consent.grant(
        ws, stored["id"], "training in our own AWS account", acknowledge_personal_data=True
    )
    with pytest.raises(PolicyDenied, match="never executed"):
        consent.import_run(ws, tmp_path, stored["id"])
    AuditLog(ws.audit_log).append("cloud.executed", subjects={"plan": stored["id"]})
    # Simulate the job's output with a local run moved out of the workspace.
    run = train_service.train(ws, version)
    out = tmp_path / run.id
    shutil.move(run.path, out)
    (out / "adapter" / "adapter_config.json").write_text("{}")
    with pytest.raises(IntegrityError, match="adapter"):
        consent.import_run(ws, out, stored["id"])
    shutil.rmtree(out)
    run = train_service.train(ws, version, seed=9)
    out = tmp_path / run.id
    shutil.move(run.path, out)
    imported = consent.import_run(ws, out, stored["id"])
    assert train_service.load_run(ws, imported).id == run.id
    events = [e.event for e in AuditLog(ws.audit_log).entries() if e.subjects.get("run") == run.id]
    assert "cloud.run_imported" in events


def test_settings_validation(workspace):
    from tests.conftest import replace_tables

    path = workspace.root / "lineage.toml"
    path.write_text(replace_tables(path.read_text(), "cloud", '[cloud]\nprovider = "aws"'))
    with pytest.raises(LineageError, match="missing"):
        consent.settings(Workspace.load(workspace.root))
