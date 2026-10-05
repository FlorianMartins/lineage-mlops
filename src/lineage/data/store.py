"""Content-addressed dataset store.

A *record* is ``{"input": str, "output": str}`` plus optional ``meta``. Its hash is the
SHA-256 of the canonical JSON of the input/output pair: two records with the same
content have the same hash wherever they come from, and ``meta`` (ids, sources) can
never make a poisoned record look different from its twin.

A *dataset version* is a set of named splits (``train``, ``heldout`` ...). Its id is
the SHA-256 over the ordered record hashes of every split, so changing, adding,
removing or reordering a single record produces a different version. Versions are
immutable: a derived dataset (for example one with canaries planted) is a new
version that names its parent.
"""

from __future__ import annotations

import csv
import json
import shutil
import tempfile
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lineage.errors import IntegrityError, LineageError
from lineage.hashing import sha256_file, sha256_json

VERSION_PREFIX = "ds-"


@dataclass(frozen=True)
class Record:
    """One training or evaluation example."""

    input: str
    output: str
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def hash(self) -> str:
        """Content hash (input and output only)."""
        return sha256_json({"input": self.input, "output": self.output})

    @property
    def id(self) -> str:
        """Human id from ``meta`` if any, else the short hash."""
        return str(self.meta.get("id") or self.hash[:12])

    def to_line(self) -> dict[str, Any]:
        """Stored form, with the hash alongside so it can be re-checked."""
        line: dict[str, Any] = {"hash": self.hash, "input": self.input, "output": self.output}
        if self.meta:
            line["meta"] = self.meta
        return line


@dataclass(frozen=True)
class Split:
    """An ordered list of records with a name."""

    name: str
    records: tuple[Record, ...]

    @property
    def hash(self) -> str:
        """Ordered hash of the split."""
        return sha256_json([r.hash for r in self.records])


@dataclass(frozen=True)
class DatasetVersion:
    """An immutable, content-addressed dataset."""

    name: str
    splits: tuple[Split, ...]
    parent: str | None = None
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def version(self) -> str:
        """The dataset version id: ``ds-`` + SHA-256 over every split in order."""
        body = {s.name: s.hash for s in self.splits}
        return VERSION_PREFIX + sha256_json(body)

    def split(self, name: str) -> Split:
        """Return a split by name."""
        for split in self.splits:
            if split.name == name:
                return split
        raise LineageError(f"dataset {self.version} has no split '{name}'")

    @property
    def split_names(self) -> list[str]:
        """Names of the splits in order."""
        return [s.name for s in self.splits]

    def manifest(self) -> dict[str, Any]:
        """Metadata stored next to the records."""
        return {
            "version": self.version,
            "name": self.name,
            "parent": self.parent,
            "splits": {s.name: {"records": len(s.records), "hash": s.hash} for s in self.splits},
            "provenance": self.provenance,
        }


# ---------------------------------------------------------------------------
# Reading raw files
# ---------------------------------------------------------------------------
def read_records(
    path: Path, input_field: str = "input", output_field: str = "output"
) -> list[Record]:
    """Read JSONL or CSV into records, keeping any other columns as ``meta``.

    Rows are taken as-is: no trimming, no deduplication, no fixing. What is in the
    file is what gets hashed, so a later check reports on the real data.
    """
    if not path.exists():
        raise LineageError(f"{path}: no such file")
    rows: Iterable[dict[str, Any]]
    if path.suffix in {".jsonl", ".ndjson"}:
        rows = _read_jsonl(path)
    elif path.suffix == ".csv":
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    else:
        raise LineageError(f"{path}: unsupported format (use .jsonl or .csv)")
    records = []
    for index, row in enumerate(rows, start=1):
        if input_field not in row or output_field not in row:
            raise LineageError(f"{path}: row {index} lacks '{input_field}' or '{output_field}'")
        meta = {k: v for k, v in row.items() if k not in {input_field, output_field}}
        records.append(Record(str(row[input_field]), str(row[output_field]), meta))
    return records


def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise LineageError(f"{path}:{lineno}: invalid JSON ({exc.msg})") from exc
            if not isinstance(row, dict):
                raise LineageError(f"{path}:{lineno}: each line must be a JSON object")
            yield row


# ---------------------------------------------------------------------------
# The store
# ---------------------------------------------------------------------------
class DatasetStore:
    """Immutable versions on disk under ``.lineage/datasets/<version>/``."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def path(self, version: str) -> Path:
        """Directory of a version."""
        if not version.startswith(VERSION_PREFIX) or "/" in version:
            raise LineageError(f"'{version}' is not a dataset version id")
        return self.root / version

    def resolve(self, ref: str) -> str:
        """Expand a unique prefix (``ds-1a2b``) to a full version id."""
        if not ref.startswith(VERSION_PREFIX):
            ref = VERSION_PREFIX + ref
        matches = [v for v in self.versions() if v.startswith(ref)]
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise LineageError(f"no dataset version matches '{ref}'")
        raise LineageError(f"'{ref}' is ambiguous ({len(matches)} versions match)")

    def exists(self, version: str) -> bool:
        """True if the version is stored."""
        return (self.path(version) / "manifest.json").exists()

    def put(self, dataset: DatasetVersion) -> tuple[str, bool]:
        """Store a version; return ``(version, created)``. Idempotent by content."""
        version = dataset.version
        target = self.path(version)
        if self.exists(version):
            return version, False
        self.root.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".tmp-", dir=self.root))
        try:
            for split in dataset.splits:
                with (staging / f"{split.name}.jsonl").open("w", encoding="utf-8") as handle:
                    for record in split.records:
                        handle.write(json.dumps(record.to_line(), ensure_ascii=False))
                        handle.write("\n")
            manifest = dataset.manifest()
            manifest["files"] = {
                f"{s.name}.jsonl": sha256_file(staging / f"{s.name}.jsonl") for s in dataset.splits
            }
            (staging / "manifest.json").write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            staging.rename(target)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        return version, True

    def get(self, version: str, *, verify: bool = True) -> DatasetVersion:
        """Load a version; by default re-hash everything and refuse on mismatch."""
        directory = self.path(version)
        manifest_path = directory / "manifest.json"
        if not manifest_path.exists():
            raise LineageError(f"unknown dataset version {version}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        splits = []
        for name in manifest["splits"]:
            file = directory / f"{name}.jsonl"
            if verify and sha256_file(file) != manifest["files"][f"{name}.jsonl"]:
                raise IntegrityError(f"{version}/{name}.jsonl was modified after ingestion")
            records = []
            for row in _read_jsonl(file):
                record = Record(row["input"], row["output"], dict(row.get("meta", {})))
                if verify and record.hash != row["hash"]:
                    raise IntegrityError(f"{version}/{name}: record {row['hash'][:12]} altered")
                records.append(record)
            splits.append(Split(name, tuple(records)))
        dataset = DatasetVersion(
            name=manifest["name"],
            splits=tuple(splits),
            parent=manifest.get("parent"),
            provenance=dict(manifest.get("provenance", {})),
        )
        if verify and dataset.version != version:
            raise IntegrityError(f"{version}: content hashes to {dataset.version}")
        return dataset

    def manifest(self, version: str) -> dict[str, Any]:
        """Raw manifest of a version."""
        data: dict[str, Any] = json.loads(
            (self.path(version) / "manifest.json").read_text(encoding="utf-8")
        )
        return data

    def versions(self) -> list[str]:
        """All stored versions."""
        if not self.root.exists():
            return []
        return sorted(p.name for p in self.root.iterdir() if (p / "manifest.json").exists())

    def write_artifact(self, version: str, name: str, content: str) -> Path:
        """Write a derived report (validation, data card) next to a version."""
        path = self.path(version) / name
        path.write_text(content, encoding="utf-8")
        return path
