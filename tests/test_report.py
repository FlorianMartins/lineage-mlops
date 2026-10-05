import json

import pytest

from lineage.audit import AuditLog
from lineage.cli import main
from lineage.registry import service as registry_service
from lineage.report import compliance
from lineage.report.history import history
from lineage.report.verify import verify_all

pytestmark = [pytest.mark.ml, pytest.mark.tools]


def failed(result):
    return {(c.area, c.subject.split(" ", 1)[-1]) for c in result.checks if not c.ok}


def test_clean_workspace_verifies(served):
    ws, _ = served
    result = verify_all(ws)
    assert result.ok, result.to_dict()["failed"]
    areas = {c.area for c in result.checks}
    assert areas == {"chain", "datasets", "runs", "registry", "deployments"}


def test_hand_edited_production_pointer_is_caught(served):
    ws, _ = served
    index_path = ws.registry / "ticket-triage" / "index.json"
    index = json.loads(index_path.read_text())
    index["versions"]["1"]["stage"] = "archived"
    index["versions"]["2"]["stage"] = "production"  # never promoted through the policy
    index_path.write_text(json.dumps(index))
    bad = {c.subject: c.detail for c in verify_all(ws).checks if not c.ok}
    assert "index says production, the log says staging" in bad["ticket-triage:2 stage"]


def test_truncated_log_is_caught_by_signed_anchors(served):
    ws, _ = served
    lines = ws.audit_log.read_text().splitlines()
    ws.audit_log.write_text("\n".join(lines[:5]) + "\n")  # still a valid chain
    result = verify_all(ws)
    assert result.to_dict()["checks"][0]["ok"]  # the chain alone looks fine...
    anchors = [c for c in result.checks if c.subject.endswith("audit anchor")]
    assert anchors and not any(c.ok for c in anchors)  # ...the signed heads are gone


def test_external_anchor_and_tampered_artefacts(served):
    ws, _ = served
    head = AuditLog(ws.audit_log).head()
    assert verify_all(ws, anchors=[head]).ok
    assert not verify_all(ws, anchors=["f" * 64]).ok
    run_dir = next(p for p in ws.runs.iterdir() if (p / "run.json").exists())
    (run_dir / "adapter" / "adapter_config.json").write_text("{}")
    deployment = next((ws.state / "deployments").glob("*.json"))
    deployment.write_text(deployment.read_text().replace('"parity"', '"parity" ', 1))
    areas = {c.area for c in verify_all(ws).checks if not c.ok}
    assert {"runs", "deployments"} <= areas


def test_history_follows_the_links(served):
    ws, _ = served
    story = history(ws, "ticket-triage:1")
    events = [e["event"] for e in story["events"]]
    for expected in (
        "data.ingested",
        "data.canaries_planted",
        "model.fetched",
        "train.finished",
        "eval.completed",
        "registry.registered",
        "registry.approved",
        "registry.promoted",
        "serve.deployed",
    ):
        assert expected in events, expected
    assert len(story["datasets"]) == 2  # canary version and its parent
    # Nothing about version 2's run leaks into version 1's story.
    other = json.loads((ws.registry / "ticket-triage" / "2" / "manifest.json").read_text())["run"]
    assert not any(other in json.dumps(e) for e in story["events"])


def test_compliance_report(served, tmp_path, monkeypatch):
    ws, _ = served
    report = compliance.build(ws, "ticket-triage:1")
    status = {c["id"]: c["status"] for c in report["controls"]}
    assert status["C10"] == status["C11"] == status["C15"] == "met"
    assert status["C14"] == "n/a"  # nothing went to the cloud
    c11 = next(c for c in report["controls"] if c["id"] == "C11")
    assert "['bob']" in c11["evidence"]
    body = {k: v for k, v in report.items() if k != "report_hash"}
    from lineage.hashing import sha256_json

    assert sha256_json(body) == report["report_hash"]

    monkeypatch.setenv("COSIGN_PASSWORD", "test")
    out = tmp_path / "report.html"
    assert main(
        ["-C", str(ws.root), "report", "compliance", "ticket-triage:1", "-o", str(out), "--sign"]
    ) in (0, 1)
    text = out.read_text()
    assert text.startswith("<!doctype html>") and "<script" not in text
    assert (tmp_path / "report.html.sig.bundle").exists()
    assert any(e.event == "report.compliance" for e in AuditLog(ws.audit_log).entries())


def test_rollback_appears_in_history_and_report(served):
    ws, _ = served
    registry_service.promote(ws, "ticket-triage:2", "production")
    registry_service.rollback(ws, None, "v2 misroutes security tickets")
    assert verify_all(ws).ok
    events = [e["event"] for e in history(ws, "ticket-triage:1")["events"]]
    assert "registry.rolled_back" in events
