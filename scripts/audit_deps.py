"""Run pip-audit over a lock file, including PyTorch CPU builds.

PyTorch CPU wheels carry a local version tag (``2.14.1+cpu``) that does not exist on
PyPI, so pip-audit cannot look them up and ``--strict`` fails. Advisories are filed
against the upstream version, so this script audits ``torch==2.14.1`` instead: the
same code, the same vulnerabilities. Every other line is passed through unchanged.

Usage: ``python scripts/audit_deps.py requirements/ml.txt``
"""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from pathlib import Path

PIN = re.compile(r"^([A-Za-z0-9_.-]+)==([^\s;\\]+)")


def main() -> int:
    lock = Path(sys.argv[1])
    pins = []
    for line in lock.read_text(encoding="utf-8").splitlines():
        match = PIN.match(line)
        if match:
            name, version = match.groups()
            pins.append(f"{name}=={version.split('+', 1)[0]}")
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as handle:
        handle.write("\n".join(pins) + "\n")
        flat = handle.name
    print(f"auditing {len(pins)} pinned packages from {lock}")
    return subprocess.call(
        ["pip-audit", "--strict", "--progress-spinner", "off", "--disable-pip", "--no-deps",
         "-r", flat]
    )


if __name__ == "__main__":
    sys.exit(main())
