"""Hyperparameter sweep with model selection on a validation split.

Choosing epochs, learning rate or seed by looking at the held-out scores turns the
held-out set into training data for the *person*: the reported score is then optimistic.
A sweep trains every combination of a small grid, scores each run on a separate
``validation`` split, and selects on that alone. The held-out split is refused as a
selection split; it is used once, by the gates, on the selected run.

Every run of the sweep is an ordinary run (tracked, logged, reproducible); the sweep
itself is one audit entry listing all of them and the one it selected.
"""

from __future__ import annotations

import itertools
import json
from dataclasses import fields
from datetime import UTC, datetime
from typing import Any

from lineage.audit import AuditLog
from lineage.data import service as data_service
from lineage.errors import LineageError, PolicyDenied
from lineage.train import service as train_service
from lineage.train.config import TrainConfig
from lineage.workspace import Workspace

ALIASES: dict[str, dict[str, tuple[str, ...]]] = {
    "target_modules": {
        "attn": ("q_proj", "k_proj", "v_proj", "o_proj"),
        "all": ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"),
    }
}


def parse_grid(items: list[str]) -> dict[str, list[Any]]:
    """``["learning_rate=5e-4,1e-3", "target_modules=attn,all"]`` -> typed values."""
    types = {f.name: type(getattr(TrainConfig(), f.name)) for f in fields(TrainConfig)}
    grid: dict[str, list[Any]] = {}
    for item in items:
        key, sep, raw = item.partition("=")
        if not sep or key not in types or key == "extra":
            raise LineageError(f"bad grid entry '{item}' (use a [train] key: name=v1,v2)")
        values: list[Any] = []
        for value in raw.split(","):
            if key in ALIASES:
                if value not in ALIASES[key]:
                    raise LineageError(f"{key} accepts {sorted(ALIASES[key])}")
                values.append(ALIASES[key][value])
            else:
                values.append(types[key](value))
        grid[key] = values
    if not grid:
        raise LineageError("give at least one --grid entry")
    return grid


def combinations(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """Cartesian product, in a stable order."""
    keys = sorted(grid)
    return [
        dict(zip(keys, values, strict=True))
        for values in itertools.product(*[grid[k] for k in keys])
    ]


def label(overrides: dict[str, Any]) -> dict[str, Any]:
    """Readable form (aliases instead of module tuples)."""
    out = {}
    for key, value in overrides.items():
        alias = next((a for a, v in ALIASES.get(key, {}).items() if v == value), None)
        out[key] = alias or value
    return out


def sweep(
    ws: Workspace, dataset_ref: str, grid: dict[str, list[Any]], split: str = "validation"
) -> dict[str, Any]:
    """Train every combination, score on ``split``, select the best."""
    from lineage.evaluate.service import split_score

    heldout = str(ws.section("eval").get("heldout_split", "heldout"))
    if split == heldout:
        raise PolicyDenied(
            f"'{split}' is the held-out split the gates use; selecting on it "
            "would leak the test set into the choice of model"
        )
    version = data_service.store_of(ws).resolve(dataset_ref)
    if split not in data_service.store_of(ws).get(version).split_names:
        raise LineageError(f"{version} has no '{split}' split to select on")
    data_service.require_trainable(ws, version)
    results = []
    for overrides in combinations(grid):
        run = train_service.train(ws, version, **overrides)
        score = split_score(ws, run.id, split)
        results.append(
            {
                "run": run.id,
                "overrides": label(overrides),
                "config_hash": run.record["config_hash"],
                **score,
            }
        )
    best = max(
        results,
        key=lambda r: (r["exact_match"], sum(r["field_accuracy"].values()), -results.index(r)),
    )
    # Runs whose interval overlaps the winner's are not shown to be worse: say so,
    # rather than pretend the ranking below the noise floor means something.
    low = best["exact_match_ci95"][0]
    for result in results:
        result["indistinguishable_from_selected"] = result["exact_match_ci95"][1] >= low
    record = {
        "dataset": version,
        "selection_split": split,
        "grid": {k: [label({k: v})[k] for v in vs] for k, vs in grid.items()},
        "results": results,
        "selected": best["run"],
        "at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    directory = ws.state / "sweeps"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"sweep-{record['at'].replace(':', '')}.json"
    path.write_text(json.dumps(record, indent=2, default=str) + "\n")
    AuditLog(ws.audit_log).append(
        "train.sweep",
        subjects={"dataset": version, "run": best["run"]},
        payload={
            "selection_split": split,
            "runs": len(results),
            "selected": best["run"],
            "results": [
                {"run": r["run"], "overrides": r["overrides"], "exact_match": r["exact_match"]}
                for r in results
            ],
        },
    )
    return record
