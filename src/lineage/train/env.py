"""Capture the environment a run executed in."""

from __future__ import annotations

import importlib.metadata
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

from lineage import __version__
from lineage.hashing import sha256_file, sha256_json

KEY_PACKAGES = (
    "torch",
    "transformers",
    "peft",
    "safetensors",
    "tokenizers",
    "mlflow",
    "numpy",
    "huggingface-hub",
)


def installed() -> dict[str, str]:
    """Every installed distribution, name -> version."""
    found: dict[str, str] = {}
    for dist in importlib.metadata.distributions():
        name = dist.metadata["Name"]
        if name:
            found[name.lower()] = dist.version
    return dict(sorted(found.items()))


def git_state(path: Path) -> dict[str, Any]:
    """Commit and dirtiness of the repository containing ``path`` (if any).

    Inside a container built from a clean checkout there is no ``.git``; the image
    records the commit it was built from in ``LINEAGE_CODE_COMMIT`` instead.
    """
    try:
        commit = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(path), "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        baked = os.environ.get("LINEAGE_CODE_COMMIT")
        if baked and baked != "unknown":
            return {"commit": baked, "dirty": False, "source": "image"}
        return {"commit": None, "dirty": None}
    return {"commit": commit, "dirty": bool(dirty)}


def capture(root: Path, lock_file: Path | None) -> dict[str, Any]:
    """Everything needed to say *where* a model came from, beyond its data and config."""
    packages = installed()
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "lineage": __version__,
        "packages": {k: packages.get(k) for k in KEY_PACKAGES},
        "packages_sha256": sha256_json(packages),
        "lock_file": str(lock_file) if lock_file else None,
        "lock_sha256": sha256_file(lock_file) if lock_file and lock_file.exists() else None,
        "code": git_state(root),
    }
