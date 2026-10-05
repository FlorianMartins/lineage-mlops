"""Where Lineage keeps its state.

A workspace is a project directory containing ``lineage.toml`` and a ``.lineage/``
state directory. All paths are derived here so no other module invents its own layout.
"""

from __future__ import annotations

import getpass
import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lineage.errors import LineageError

CONFIG_NAME = "lineage.toml"
STATE_DIR = ".lineage"


@dataclass(frozen=True)
class Workspace:
    """Resolved paths and configuration for one project."""

    root: Path
    config: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, root: Path | None = None) -> Workspace:
        """Load the workspace rooted at ``root`` (default: ``$LINEAGE_HOME`` or cwd)."""
        base = root or Path(os.environ.get("LINEAGE_HOME", Path.cwd()))
        base = base.resolve()
        config_path = base / CONFIG_NAME
        config: dict[str, Any] = {}
        if config_path.exists():
            with config_path.open("rb") as handle:
                config = tomllib.load(handle)
        return cls(root=base, config=config)

    # -- state layout -------------------------------------------------------
    @property
    def state(self) -> Path:
        """The ``.lineage`` directory."""
        return self.root / STATE_DIR

    @property
    def audit_log(self) -> Path:
        """The single hash-chained audit log every step writes to."""
        return self.state / "audit" / "log.jsonl"

    @property
    def datasets(self) -> Path:
        """Content-addressed dataset store."""
        return self.state / "datasets"

    @property
    def canaries(self) -> Path:
        """Secret canary lists, kept outside the datasets they were planted in."""
        return self.state / "canaries"

    @property
    def models(self) -> Path:
        """Pinned and verified base-model snapshots."""
        return self.state / "models"

    @property
    def runs(self) -> Path:
        """Training run outputs."""
        return self.state / "runs"

    @property
    def registry(self) -> Path:
        """Model registry (versions, stages, signatures)."""
        return self.state / "registry"

    @property
    def proposals(self) -> Path:
        """Retraining proposals raised by drift monitoring (never auto-executed)."""
        return self.state / "proposals"

    @property
    def consents(self) -> Path:
        """Recorded consents for actions that send data off the machine."""
        return self.state / "consents"

    def section(self, name: str) -> dict[str, Any]:
        """Return a top-level table of ``lineage.toml`` (empty if absent)."""
        value = self.config.get(name, {})
        if not isinstance(value, dict):
            raise LineageError(f"[{name}] in {CONFIG_NAME} must be a table")
        return value

    def resolve(self, path: str | Path) -> Path:
        """Resolve a path from the config relative to the workspace root."""
        candidate = Path(path)
        return candidate if candidate.is_absolute() else self.root / candidate


def current_actor() -> str:
    """Who is acting: ``$LINEAGE_ACTOR`` if set (CI sets it), else the OS user."""
    actor = os.environ.get("LINEAGE_ACTOR")
    if actor:
        return actor
    try:
        return getpass.getuser()
    except (KeyError, OSError):  # pragma: no cover - containers without a passwd entry
        return "unknown"
