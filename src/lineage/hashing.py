"""Canonical hashing.

Everything that Lineage addresses by content (records, datasets, configs, manifests,
audit entries) goes through these helpers, so that "the same thing" always produces
the same digest regardless of key order, whitespace or platform.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

_CHUNK = 1 << 20


def canonical_json(value: Any) -> bytes:
    """Serialise ``value`` deterministically (sorted keys, no whitespace, UTF-8)."""
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    """Hex SHA-256 of raw bytes."""
    return hashlib.sha256(data).hexdigest()


def sha256_json(value: Any) -> str:
    """Hex SHA-256 of the canonical JSON form of ``value``."""
    return sha256_bytes(canonical_json(value))


def sha256_file(path: Path) -> str:
    """Hex SHA-256 of a file, streamed so multi-gigabyte weights do not load into RAM."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def short(digest: str, length: int = 12) -> str:
    """Abbreviate a digest for display; never use the short form for verification."""
    return digest.split(":", 1)[-1][:length]
