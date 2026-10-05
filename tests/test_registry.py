import json
import shutil

import pytest

from lineage.audit import AuditLog
from lineage.cli import main
from lineage.errors import IntegrityError, LineageError, PolicyDenied
from lineage.registry import service
from lineage.registry.store import Registry
from lineage.workspace import Workspace
from tests.conftest import replace_tables

pytestmark = [
    pytest.mark.ml,
    pytest.mark.tools,
    pytest.mark.skipif(
        not (shutil.which("opa") and shutil.which("cosign")), reason="needs opa and cosign on PATH"
    ),
]


def as_actor(monkeypatch, name):
    monkeypatch.setenv("LINEAGE_ACTOR", name)


def test_register_signs_and_writes_a_valid_mlbom(promotable):
    from cyclonedx.schema import SchemaVersion
    from cyclonedx.validation.json import JsonStrictValidator

    ws, runs = promotable
    number = service.register(ws, runs[0])
    registry = Registry(ws)
    path = registry.path(number)
    assert registry.verify(number)["ok"]
    bom = (path / "mlbom.cdx.json").read_text()
    assert JsonStrictValidator(SchemaVersion.V1_6).validate_str(bom) is None
    doc = json.loads(bom)
    assert doc["metadata"]["component"]["type"] == "machine-learning-model"
    assert {c["type"] for c in doc["components"]} >= {"machine-learning-model", "data", "platform"}
    manifest = registry.manifest(number)
    assert AuditLog(ws.audit_log).contains(manifest["audit_head"])
    assert "adapter/adapter_model.safetensors" in manifest["files"]
    with pytest.raises(LineageError, match="already"):
        service.register(ws, runs[0])


def test_full_promotion_flow_with_approvals_and_rollback(promotable, monkeypatch):
    ws, runs = promotable
    v1 = service.register(ws, runs[0])
    with pytest.raises(PolicyDenied, match="denied"):
        service.promote(ws, f"ticket-triage:{v1}", "production")  # no skipping staging
    service.promote(ws, f"ticket-triage:{v1}", "staging")
    with pytest.raises(PolicyDenied) as denied:
        service.promote(ws, f"ticket-triage:{v1}", "production")
    assert "independent approval" in denied.value.reasons[0]
    service.approve(ws, f"ticket-triage:{v1}", "I registered it, I approve it")  # alice: registrant
    as_actor(monkeypatch, "trainer")
    service.approve(ws, f"ticket-triage:{v1}", "I trained it and it looks fine")
    with pytest.raises(PolicyDenied):
        service.promote(ws, f"ticket-triage:{v1}", "production")
    as_actor(monkeypatch, "bob")
    service.approve(ws, f"ticket-triage:{v1}", "reviewed the evaluation report")
    service.promote(ws, f"ticket-triage:{v1}", "production")
    registry = Registry(ws)
    assert registry.production()["version"] == v1

    as_actor(monkeypatch, "alice")
    v2 = service.register(ws, runs[1])
    service.promote(ws, f"ticket-triage:{v2}", "staging")
    as_actor(monkeypatch, "bob")
    service.approve(ws, f"ticket-triage:{v2}", "reviewed the evaluation report")
    service.promote(ws, f"ticket-triage:{v2}", "production")
    stages = {v["version"]: v["stage"] for v in registry.versions()}
    assert stages == {v1: "archived", v2: "production"}

    result = service.rollback(ws, None, "v2 misroutes security tickets")
    assert result == {
        "model": "ticket-triage",
        "version": v1,
        "from": "archived",
        "to": "production",
        "replaced": v2,
    }
    stages = {v["version"]: v["stage"] for v in registry.versions()}
    assert stages == {v1: "production", v2: "rolled_back"}
    with pytest.raises(LineageError, match="no earlier production"):
        service.rollback(ws, None, "rolling back twice must not bring v2 back")

    events = [e.event for e in AuditLog(ws.audit_log).entries()]
    assert events.count("registry.promote_denied") == 3
    assert "registry.rolled_back" in events
    assert AuditLog(ws.audit_log).verify().ok


def test_tampering_blocks_promotion_and_is_logged(promotable):
    ws, runs = promotable
    number = service.register(ws, runs[0])
    registry = Registry(ws)
    adapter = registry.path(number) / "adapter" / "adapter_model.safetensors"
    adapter.write_bytes(adapter.read_bytes()[:-1] + b"\x00")
    result = registry.verify(number)
    assert not result["manifest_intact"] and result["signature_verified"]
    with pytest.raises(PolicyDenied) as denied:
        service.promote(ws, f"ticket-triage:{number}", "staging")
    assert "no longer match" in " ".join(denied.value.reasons)
    with pytest.raises(IntegrityError):
        service.approve(ws, f"ticket-triage:{number}", "approving a tampered model")


def test_forged_manifest_breaks_the_signature(promotable):
    ws, runs = promotable
    number = service.register(ws, runs[0])
    registry = Registry(ws)
    manifest_path = registry.path(number) / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["eval_passed"] = True
    manifest["registered_by"] = "someone-else"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    result = registry.verify(number)
    assert not result["signature_verified"] and not result["provenance_verified"]


def test_failed_gates_cannot_be_promoted(promotable):
    ws, runs = promotable
    text = (ws.root / "lineage.toml").read_text()
    text = replace_tables(text, "gates.quality", "[gates.quality]\nmin_exact_match = 0.99")
    (ws.root / "lineage.toml").write_text(text)
    from lineage.evaluate import service as eval_service

    ws = Workspace.load(ws.root)
    eval_service.evaluate(ws, runs[0])
    number = service.register(ws, runs[0])
    with pytest.raises(PolicyDenied) as denied:
        service.promote(ws, f"ticket-triage:{number}", "staging")
    assert "evaluation gate 'quality' failed" in denied.value.reasons


def test_policy_fails_closed_without_opa(promotable, monkeypatch):
    ws, runs = promotable
    number = service.register(ws, runs[0])
    real_which = shutil.which
    monkeypatch.setattr(shutil, "which", lambda name: None if name == "opa" else real_which(name))
    with pytest.raises(PolicyDenied, match="fail closed"):
        service.promote(ws, f"ticket-triage:{number}", "staging")


def test_cli(promotable, capsys):
    ws, runs = promotable
    root = str(ws.root)
    assert main(["-C", root, "registry", "register", runs[0]]) == 0
    assert main(["-C", root, "registry", "verify", "ticket-triage:1"]) == 0
    assert main(["-C", root, "registry", "promote", "ticket-triage:1", "--to", "production"]) == 1
    assert "not an allowed transition" in capsys.readouterr().err
    assert main(["-C", root, "--json", "registry", "list"]) == 0
    assert '"stage": "candidate"' in capsys.readouterr().out
