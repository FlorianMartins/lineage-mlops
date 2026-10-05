import json
import os
import pickle
import struct

import pytest

from lineage.audit import AuditLog
from lineage.errors import IntegrityError, LineageError, PolicyDenied
from lineage.supply import licence, weights
from lineage.supply.models import ModelPin, fetch, pin_block, verify_snapshot
from tests.conftest import TINY_REVISION, FakeHub, pin_tiny


def write_safetensors(path, header, data=b""):
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + data)


class Exploit:
    def __reduce__(self):
        return (os.system, ("echo pwned",))


# -- weights -----------------------------------------------------------------
def test_valid_safetensors(tmp_path):
    path = tmp_path / "m.safetensors"
    write_safetensors(
        path, {"w": {"dtype": "F32", "shape": [2, 2], "data_offsets": [0, 16]}}, b"\0" * 16
    )
    assert weights.check_safetensors(path) == []


@pytest.mark.parametrize(
    ("header", "data", "problem"),
    [
        ({"w": {"dtype": "F32", "shape": [2, 2], "data_offsets": [0, 64]}}, b"\0" * 16, "outside"),
        (
            {"w": {"dtype": "F32", "shape": [3], "data_offsets": [0, 16]}},
            b"\0" * 16,
            "does not match",
        ),
        (
            {
                "a": {"dtype": "U8", "shape": [8], "data_offsets": [0, 8]},
                "b": {"dtype": "U8", "shape": [8], "data_offsets": [4, 12]},
            },
            b"\0" * 12,
            "overlap",
        ),
        ({"w": {"dtype": "X9", "shape": [1], "data_offsets": [0, 1]}}, b"\0", "unknown dtype"),
        ({"w": {"shape": [1]}}, b"", "malformed"),
    ],
)
def test_invalid_safetensors(tmp_path, header, data, problem):
    path = tmp_path / "m.safetensors"
    write_safetensors(path, header, data)
    assert any(problem in p for p in weights.check_safetensors(path))


def test_lying_header_length(tmp_path):
    path = tmp_path / "m.safetensors"
    path.write_bytes(struct.pack("<Q", 10**12) + b"{}")
    assert "exceeds" in weights.check_safetensors(path)[0]


def test_pickle_scan_flags_code_execution_without_running_it(tmp_path):
    path = tmp_path / "pytorch_model.bin"
    path.write_bytes(pickle.dumps(Exploit()))
    scan = weights.scan_pickle(path)
    assert any(g.endswith(".system") for g in scan.dangerous)
    assert not (tmp_path / "pwned").exists()


def test_renamed_pickle_is_still_a_pickle(tmp_path):
    (tmp_path / "weights.safe").write_bytes(pickle.dumps({"a": 1}))
    write_safetensors(tmp_path / "model.safetensors", {})
    inspection = weights.inspect(tmp_path)
    assert inspection.pickles == ["weights.safe"] and not inspection.ok


def test_torch_checkpoint_scan_lists_rebuild_functions(tmp_path):
    torch = pytest.importorskip("torch")
    path = tmp_path / "ckpt.pt"
    torch.save({"w": torch.zeros(2)}, path)
    scan = weights.scan_pickle(path)
    assert any("_rebuild_tensor" in g for g in scan.globals) and not scan.dangerous
    assert weights.is_pickle(path)


# -- licence -----------------------------------------------------------------
@pytest.mark.parametrize(
    ("lic", "use", "verdict"),
    [
        ("apache-2.0", "commercial", "allow"),
        ("MIT", "redistribution", "allow"),
        ("llama3.1", "commercial", "conditional"),
        ("gemma", "internal", "conditional"),
        ("cc-by-nc-4.0", "commercial", "deny"),
        ("cc-by-nc-4.0", "research", "allow"),
        ("research-only", "internal", "deny"),
        ("some-custom-licence", "research", "deny"),
        (None, "research", "deny"),
    ],
)
def test_licence_table(lic, use, verdict):
    assert licence.check(lic, use).verdict == verdict


def test_relicensed_model_is_denied():
    decision = licence.check("cc-by-nc-4.0", "internal", declared="apache-2.0")
    assert decision.verdict == "deny" and "re-review" in decision.reasons[0]


def test_unknown_use_is_an_error():
    with pytest.raises(LineageError):
        licence.check("mit", "everything")


# -- pin and fetch -------------------------------------------------------------
def _pinned_workspace(workspace, tiny_model_dir, hub=None, use="commercial"):
    hub = hub or FakeHub(tiny_model_dir)
    return pin_tiny(workspace.root, hub, use), hub, None


def test_revision_must_be_a_commit_sha():
    with pytest.raises(PolicyDenied, match=r"commit SHA"):
        ModelPin.from_config({"repo": "x", "revision": "main", "files": {"a": "0" * 64}})
    with pytest.raises(PolicyDenied):
        pin_block(FakeHub(None), "x", "v1.0", None)


def test_pin_skips_pickles_and_requires_safetensors(tmp_path, tiny_model_dir):
    import shutil

    repo = tmp_path / "repo"
    shutil.copytree(tiny_model_dir, repo)
    (repo / "training_args.bin").write_bytes(pickle.dumps({"lr": 1}))
    block, skipped = pin_block(FakeHub(repo), "x", TINY_REVISION, tmp_path / "dl")
    assert "training_args.bin (pickle)" in skipped and "model.safetensors" in block
    (repo / "model.safetensors").unlink()
    with pytest.raises(PolicyDenied, match="no safetensors"):
        pin_block(FakeHub(repo), "x", TINY_REVISION, tmp_path / "dl2")


def test_fetch_verifies_and_logs(workspace, tiny_model_dir):
    ws, hub, _ = _pinned_workspace(workspace, tiny_model_dir)
    path = fetch(ws, hub)
    assert (path / "manifest.json").exists()
    pin, _ = verify_snapshot(ws)
    events = [e for e in AuditLog(ws.audit_log).entries() if e.event == "model.fetched"]
    assert events and events[0].subjects["base_model"] == pin.ref
    assert events[0].payload["licence"]["verdict"] == "allow"


def test_fetch_refuses_a_file_whose_hash_changed(workspace, tiny_model_dir, tmp_path):
    import shutil

    ws, _, _ = _pinned_workspace(workspace, tiny_model_dir)
    tampered = tmp_path / "tampered"
    shutil.copytree(tiny_model_dir, tampered)
    with (tampered / "config.json").open("a") as handle:
        handle.write(" ")
    with pytest.raises(IntegrityError, match=r"config\.json"):
        fetch(ws, FakeHub(tampered))
    rejected = [e for e in AuditLog(ws.audit_log).entries() if e.event == "model.rejected"]
    assert rejected[0].payload["reason"] == "hash_mismatch"
    assert not any(ws.models.glob("*/manifest.json"))


def test_fetch_refuses_a_non_commercial_licence_for_commercial_use(workspace, tiny_model_dir):
    hub = FakeHub(tiny_model_dir, licence="cc-by-nc-4.0")
    ws, _, _ = _pinned_workspace(workspace, tiny_model_dir, hub=hub)
    with pytest.raises(PolicyDenied, match="licence"):
        fetch(ws, hub)


def test_conditional_licence_needs_recorded_acceptance(workspace, tiny_model_dir):
    from lineage.supply.models import accept_licence

    hub = FakeHub(tiny_model_dir, licence="llama3.1")
    ws, _, _ = _pinned_workspace(workspace, tiny_model_dir, hub=hub)
    fetch(ws, hub)
    with pytest.raises(PolicyDenied, match="conditions"):
        verify_snapshot(ws)
    accept_licence(ws, "legal reviewed the Llama licence (ticket LEG-42)")
    verify_snapshot(ws)


def test_verify_detects_tampering_and_extra_files(workspace, tiny_model_dir):
    ws, hub, _ = _pinned_workspace(workspace, tiny_model_dir)
    path = fetch(ws, hub)
    (path / "extra.pkl").write_bytes(pickle.dumps(1))
    with pytest.raises(PolicyDenied, match="outside the pin"):
        verify_snapshot(ws)
    (path / "extra.pkl").unlink()
    with (path / "model.safetensors").open("r+b") as handle:
        handle.seek(-1, 2)
        handle.write(b"\x01")
    with pytest.raises(IntegrityError):
        verify_snapshot(ws)
