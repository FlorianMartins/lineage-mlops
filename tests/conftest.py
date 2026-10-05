from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from lineage.workspace import Workspace

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "triage"


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Workspace:
    """A fresh copy of the example workspace (config + data, no state)."""
    shutil.copy(EXAMPLE / "lineage.toml", tmp_path / "lineage.toml")
    shutil.copytree(EXAMPLE / "data", tmp_path / "data")
    monkeypatch.setenv("LINEAGE_ACTOR", "tester")
    monkeypatch.chdir(tmp_path)
    return Workspace.load(tmp_path)


@pytest.fixture
def clean_version(workspace: Workspace) -> str:
    from lineage.data import service

    version, _ = service.ingest(
        workspace,
        "triage",
        {
            "train": workspace.root / "data/train.jsonl",
            "heldout": workspace.root / "data/heldout.jsonl",
        },
    )
    return version


@pytest.fixture
def poisoned_version(workspace: Workspace) -> str:
    from lineage.data import service

    version, _ = service.ingest(
        workspace,
        "triage-poisoned",
        {
            "train": workspace.root / "data/poisoned.jsonl",
            "heldout": workspace.root / "data/heldout.jsonl",
        },
    )
    return version
