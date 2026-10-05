"""The hash-chained audit log.

Every step of the lifecycle appends one entry. Each entry embeds the hash of the
previous one, so editing, deleting or reordering any past entry breaks every hash
after it. ``verify`` recomputes the whole chain.

A chain alone cannot detect that its *tail* was cut off. Two things cover that:
signed registry manifests embed the audit head at signing time (so a chain that no
longer contains that head was truncated), and ``lineage audit head`` prints the
current head so it can be published somewhere else (a ticket, a commit, a log sink).

Entries name the things they are about in ``subjects`` (``dataset``, ``run``,
``model``, ``base_model`` ...). ``lineage history`` follows those links to rebuild
the full story of a model.
"""

from __future__ import annotations

import fcntl
import json
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from lineage.hashing import sha256_json
from lineage.workspace import current_actor

GENESIS = "0" * 64


def utc_now() -> str:
    """Current UTC time, ISO 8601 with second precision."""
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class Entry:
    """One audit entry, as stored."""

    seq: int
    ts: str
    actor: str
    event: str
    subjects: dict[str, str]
    payload: dict[str, Any]
    prev: str
    hash: str

    def body(self) -> dict[str, Any]:
        """Everything the hash covers (all fields but the hash itself)."""
        return {
            "seq": self.seq,
            "ts": self.ts,
            "actor": self.actor,
            "event": self.event,
            "subjects": self.subjects,
            "payload": self.payload,
            "prev": self.prev,
        }

    def to_dict(self) -> dict[str, Any]:
        """Stored form."""
        return {**self.body(), "hash": self.hash}

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Entry:
        """Parse a stored entry (no verification; see :func:`AuditLog.verify`)."""
        return cls(
            seq=int(raw["seq"]),
            ts=str(raw["ts"]),
            actor=str(raw["actor"]),
            event=str(raw["event"]),
            subjects={str(k): str(v) for k, v in dict(raw.get("subjects", {})).items()},
            payload=dict(raw.get("payload", {})),
            prev=str(raw["prev"]),
            hash=str(raw["hash"]),
        )


@dataclass
class VerifyResult:
    """Outcome of a chain verification."""

    entries: int = 0
    head: str = GENESIS
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when every entry links and hashes correctly."""
        return not self.errors


class AuditLog:
    """Append-only, hash-chained JSONL log."""

    def __init__(self, path: Path, clock: Callable[[], str] = utc_now) -> None:
        self.path = path
        self._clock = clock

    def append(
        self,
        event: str,
        *,
        subjects: dict[str, str] | None = None,
        payload: dict[str, Any] | None = None,
        actor: str | None = None,
    ) -> Entry:
        """Append one entry and return it.

        The file is locked for the read-tail/append so two processes cannot both
        link to the same previous entry.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                handle.seek(0)
                last = _last_entry(handle.read().splitlines())
                seq = last.seq + 1 if last else 0
                prev = last.hash if last else GENESIS
                draft = Entry(
                    seq=seq,
                    ts=self._clock(),
                    actor=actor or current_actor(),
                    event=event,
                    subjects=dict(subjects or {}),
                    payload=dict(payload or {}),
                    prev=prev,
                    hash="",
                )
                entry = Entry(**{**draft.body(), "hash": sha256_json(draft.body())})
                handle.seek(0, 2)
                handle.write(json.dumps(entry.to_dict(), sort_keys=True, ensure_ascii=False))
                handle.write("\n")
                handle.flush()
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)
        return entry

    def entries(self) -> Iterator[Entry]:
        """Iterate stored entries in order (unverified)."""
        if not self.path.exists():
            return
        with self.path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield Entry.from_dict(json.loads(line))

    def head(self) -> str:
        """Hash of the last entry (``GENESIS`` for an empty log)."""
        last = None
        for last in self.entries():  # noqa: B007 - we want the final one
            pass
        return last.hash if last else GENESIS

    def contains(self, digest: str) -> bool:
        """True if an entry with this hash exists in the chain."""
        return any(entry.hash == digest for entry in self.entries())

    def verify(self) -> VerifyResult:
        """Recompute every hash and link; report every break, not just the first."""
        result = VerifyResult()
        if not self.path.exists():
            return result
        expected_prev = GENESIS
        expected_seq = 0
        with self.path.open(encoding="utf-8") as handle:
            for lineno, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    entry = Entry.from_dict(json.loads(line))
                except (ValueError, KeyError, TypeError) as exc:
                    result.errors.append(f"line {lineno}: unparseable entry ({exc})")
                    expected_seq += 1
                    continue
                if entry.seq != expected_seq:
                    result.errors.append(
                        f"line {lineno}: seq {entry.seq}, expected {expected_seq} "
                        "(entry inserted or removed)"
                    )
                if entry.prev != expected_prev:
                    result.errors.append(
                        f"seq {entry.seq}: prev does not match the previous entry's hash"
                    )
                recomputed = sha256_json(entry.body())
                if recomputed != entry.hash:
                    result.errors.append(f"seq {entry.seq}: content does not match its hash")
                expected_prev = entry.hash
                expected_seq = entry.seq + 1
                result.entries += 1
                result.head = entry.hash
        return result


def _last_entry(lines: list[str]) -> Entry | None:
    for line in reversed(lines):
        if line.strip():
            return Entry.from_dict(json.loads(line))
    return None
