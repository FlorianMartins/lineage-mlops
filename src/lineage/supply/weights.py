"""Weight-file policy: safetensors only.

Pickle-based formats (``.bin``, ``.pt``, ``.ckpt``, ``.pkl`` ...) execute code when they
are loaded, so a model file is a program. Lineage never loads them. ``inspect`` reports
what a snapshot contains; ``scan_pickle`` reads a pickle's opcodes *without executing
them* and lists the callables it would import, which is what you want to see before
deciding whether a file is merely unusual or actively malicious.

Safetensors files are not trusted blindly either: the header is parsed and every
tensor's byte range must lie inside the file and not overlap another one.
"""

from __future__ import annotations

import itertools
import json
import pickletools
import struct
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PICKLE_SUFFIXES = {".bin", ".pt", ".pth", ".ckpt", ".pkl", ".pickle", ".joblib", ".npy"}
SAFE_SUFFIXES = {".safetensors"}
# Metadata files a snapshot legitimately carries.
DATA_SUFFIXES = {".json", ".txt", ".model", ".md", ".jinja", ".tiktoken"}

# Imports that turn unpickling into code execution or I/O.
DANGEROUS_GLOBALS = {
    "os",
    "posix",
    "nt",
    "subprocess",
    "sys",
    "socket",
    "shutil",
    "runpy",
    "importlib",
    "pty",
    "ctypes",
    "webbrowser",
    "requests",
    "urllib",
    "http",
    "pickle",
    "marshal",
    "code",
    "codeop",
    "commands",
}
DANGEROUS_CALLS = {
    "builtins.eval",
    "builtins.exec",
    "builtins.compile",
    "builtins.open",
    "builtins.__import__",
    "builtins.getattr",
    "__builtin__.eval",
    "__builtin__.exec",
}
_DTYPE_BYTES = {
    "F64": 8,
    "F32": 4,
    "F16": 2,
    "BF16": 2,
    "I64": 8,
    "I32": 4,
    "I16": 2,
    "I8": 1,
    "U8": 1,
    "BOOL": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "U16": 2,
    "U32": 4,
    "U64": 8,
}
_MAX_HEADER = 100 * 1024 * 1024


@dataclass
class PickleScan:
    """What a pickle would import if it were loaded."""

    path: str
    globals: list[str] = field(default_factory=list)
    dangerous: list[str] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Report form."""
        return {
            "path": self.path,
            "globals": self.globals,
            "dangerous": self.dangerous,
            "error": self.error,
        }


@dataclass
class Inspection:
    """Classification of every file in a snapshot."""

    safetensors: list[str] = field(default_factory=list)
    pickles: list[str] = field(default_factory=list)
    data: list[str] = field(default_factory=list)
    unknown: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """Loadable under the policy: no pickle, no unknown file, valid safetensors."""
        return not (self.pickles or self.unknown or self.problems) and bool(self.safetensors)


def is_pickle(path: Path) -> bool:
    """Suffix *or* content says pickle (a renamed pickle is still a pickle)."""
    if path.suffix in PICKLE_SUFFIXES:
        return True
    with path.open("rb") as handle:
        head = handle.read(4)
    # Protocol 2+ pickles start with PROTO (0x80); torch zip archives with PK.
    if head[:1] == b"\x80" and len(head) > 1 and head[1] in range(2, 6):
        return True
    return head[:2] == b"PK" and _zip_has_pickle(path)


def _zip_has_pickle(path: Path) -> bool:
    try:
        with zipfile.ZipFile(path) as archive:
            return any(name.endswith(".pkl") for name in archive.namelist())
    except zipfile.BadZipFile:
        return False


def check_safetensors(path: Path) -> list[str]:
    """Validate a safetensors header and tensor layout; return problems (empty = valid)."""
    size = path.stat().st_size
    with path.open("rb") as handle:
        raw = handle.read(8)
        if len(raw) < 8:
            return ["file shorter than the 8-byte header length"]
        (header_len,) = struct.unpack("<Q", raw)
        if header_len > min(size - 8, _MAX_HEADER):
            return [f"header length {header_len} exceeds file or limit"]
        try:
            header = json.loads(handle.read(header_len))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            return [f"header is not valid JSON ({exc})"]
    if not isinstance(header, dict):
        return ["header is not a JSON object"]
    data_len = size - 8 - header_len
    spans = []
    problems = []
    for name, info in header.items():
        if name == "__metadata__":
            continue
        try:
            start, end = (int(x) for x in info["data_offsets"])
            dtype = str(info["dtype"])
            shape = [int(x) for x in info["shape"]]
        except (KeyError, TypeError, ValueError):
            problems.append(f"tensor '{name}': malformed entry")
            continue
        if not 0 <= start <= end <= data_len:
            problems.append(f"tensor '{name}': byte range outside the file")
            continue
        expected = _DTYPE_BYTES.get(dtype)
        count = 1
        for dim in shape:
            count *= dim
        if expected is None:
            problems.append(f"tensor '{name}': unknown dtype {dtype}")
        elif count * expected != end - start:
            problems.append(f"tensor '{name}': shape {shape} does not match its byte range")
        spans.append((start, end, name))
    spans.sort()
    for (_, end_a, a), (start_b, _, b) in itertools.pairwise(spans):
        if start_b < end_a:
            problems.append(f"tensors '{a}' and '{b}' overlap")
    return problems


def inspect(directory: Path) -> Inspection:
    """Classify every file under ``directory``."""
    result = Inspection()
    for path in sorted(p for p in directory.rglob("*") if p.is_file()):
        rel = str(path.relative_to(directory))
        if path.name == "manifest.json" and path.parent == directory:
            continue
        if path.suffix in SAFE_SUFFIXES:
            result.safetensors.append(rel)
            result.problems += [f"{rel}: {p}" for p in check_safetensors(path)]
        elif is_pickle(path):
            result.pickles.append(rel)
        elif path.suffix in DATA_SUFFIXES or path.name in {"LICENSE", "NOTICE", "Modelfile"}:
            result.data.append(rel)
        else:
            result.unknown.append(rel)
    return result


def scan_pickle(path: Path) -> PickleScan:
    """List the globals a pickle imports, without unpickling it."""
    result = PickleScan(str(path))
    try:
        if zipfile.is_zipfile(path):
            with zipfile.ZipFile(path) as archive:
                for name in archive.namelist():
                    if name.endswith(".pkl"):
                        _scan_stream(archive.read(name), result)
        else:
            _scan_stream(path.read_bytes(), result)
    except Exception as exc:  # a malformed pickle is a finding, not a crash
        result.error = f"{type(exc).__name__}: {exc}"
    result.globals = sorted(set(result.globals))
    result.dangerous = sorted(set(result.dangerous))
    return result


def _scan_stream(data: bytes, result: PickleScan) -> None:
    recent_strings: list[str] = []
    for opcode, arg, _ in pickletools.genops(data):
        if opcode.name in {
            "SHORT_BINUNICODE",
            "BINUNICODE",
            "UNICODE",
            "STRING",
            "BINSTRING",
            "SHORT_BINSTRING",
        }:
            recent_strings.append(str(arg))
            recent_strings = recent_strings[-2:]
        elif opcode.name in {"GLOBAL", "INST"}:
            module, _, name = str(arg).partition(" ")
            _record(f"{module}.{name}", result)
        elif opcode.name == "STACK_GLOBAL" and len(recent_strings) == 2:
            _record(".".join(recent_strings), result)


def _record(qualified: str, result: PickleScan) -> None:
    result.globals.append(qualified)
    root = qualified.split(".", 1)[0]
    if root in DANGEROUS_GLOBALS or qualified in DANGEROUS_CALLS:
        result.dangerous.append(qualified)
