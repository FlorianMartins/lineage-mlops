import json
import threading

from lineage.audit import GENESIS, AuditLog
from lineage.hashing import canonical_json, sha256_json


def _log(tmp_path):
    return AuditLog(tmp_path / "log.jsonl", clock=lambda: "2026-10-05T00:00:00Z")


def _lines(log):
    return log.path.read_text().splitlines()


def test_canonical_json_ignores_key_order():
    assert canonical_json({"b": 1, "a": "é"}) == canonical_json({"a": "é", "b": 1})
    assert sha256_json({"b": 1, "a": 2}) == sha256_json({"a": 2, "b": 1})


def test_empty_log_verifies_with_genesis_head(tmp_path):
    log = _log(tmp_path)
    result = log.verify()
    assert result.ok and result.entries == 0 and result.head == GENESIS


def test_entries_chain_to_each_other(tmp_path):
    log = _log(tmp_path)
    first = log.append("a", subjects={"dataset": "ds-1"}, actor="alice")
    second = log.append("b", payload={"n": 1}, actor="bob")
    assert first.prev == GENESIS and second.prev == first.hash and second.seq == 1
    result = log.verify()
    assert result.ok and result.entries == 2 and result.head == second.hash == log.head()
    assert log.contains(first.hash)


def test_editing_an_entry_is_detected(tmp_path):
    log = _log(tmp_path)
    for i in range(3):
        log.append("step", payload={"i": i})
    lines = _lines(log)
    entry = json.loads(lines[1])
    entry["payload"]["i"] = 99
    lines[1] = json.dumps(entry)
    log.path.write_text("\n".join(lines) + "\n")
    result = log.verify()
    assert not result.ok
    assert any("seq 1: content does not match" in e for e in result.errors)


def test_editing_and_rehashing_still_breaks_the_next_link(tmp_path):
    log = _log(tmp_path)
    for i in range(3):
        log.append("step", payload={"i": i})
    lines = _lines(log)
    entry = json.loads(lines[1])
    entry["payload"]["i"] = 99
    entry["hash"] = sha256_json({k: v for k, v in entry.items() if k != "hash"})
    lines[1] = json.dumps(entry)
    log.path.write_text("\n".join(lines) + "\n")
    errors = log.verify().errors
    assert any("seq 2: prev does not match" in e for e in errors)


def test_deleting_and_reordering_are_detected(tmp_path):
    log = _log(tmp_path)
    for i in range(4):
        log.append("step", payload={"i": i})
    lines = _lines(log)
    log.path.write_text("\n".join([lines[0], lines[2], lines[3]]) + "\n")
    assert not log.verify().ok
    log.path.write_text("\n".join([lines[0], lines[2], lines[1], lines[3]]) + "\n")
    assert not log.verify().ok


def test_garbage_line_is_reported_not_crashing(tmp_path):
    log = _log(tmp_path)
    log.append("step")
    with log.path.open("a") as handle:
        handle.write("{not json\n")
    result = log.verify()
    assert not result.ok and "unparseable" in result.errors[0]


def test_concurrent_appends_keep_a_single_chain(tmp_path):
    log = _log(tmp_path)
    threads = [
        threading.Thread(target=lambda: [log.append("t") for _ in range(20)]) for _ in range(5)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    result = log.verify()
    assert result.ok and result.entries == 100
