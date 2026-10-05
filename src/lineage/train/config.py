"""Training configuration and its hash.

The config hash covers everything that determines the result: hyperparameters, the
dataset version, the base model revision, the task contract and the dependency lock.
Two runs with the same hash on the same machine produce the same adapter, byte for byte
(``tests/test_train.py`` checks it); a run whose hash differs is a different experiment.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, replace
from typing import Any

from lineage.errors import LineageError
from lineage.hashing import sha256_json


@dataclass(frozen=True)
class TrainConfig:
    """Hyperparameters (all have CPU-friendly defaults for a ~100M model)."""

    epochs: int = 2
    batch_size: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    warmup_ratio: float = 0.05
    max_length: int = 160
    seed: int = 1234
    threads: int = 4
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
    # "completion": loss on the answer only (the prompt is context, not something to
    # learn by heart). "full": loss on every token, which is what makes a model
    # memorise its inputs; kept to demonstrate the memorisation gate.
    loss_on: str = "completion"
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_config(cls, section: dict[str, Any], **overrides: Any) -> TrainConfig:
        """Build from ``[train]``; unknown keys are an error, not silently ignored."""
        known = {f.name for f in fields(cls)} - {"extra"}
        values = {k: v for k, v in section.items() if k != "lock_file"}
        unknown = set(values) - known
        if unknown:
            raise LineageError(f"[train] has unknown keys: {sorted(unknown)}")
        if "target_modules" in values:
            values["target_modules"] = tuple(values["target_modules"])
        config = replace(cls(), **values)
        clean = {k: v for k, v in overrides.items() if v is not None}
        config = replace(config, **clean)
        if config.loss_on not in {"completion", "full"}:
            raise LineageError("[train].loss_on must be 'completion' or 'full'")
        return config

    def to_dict(self) -> dict[str, Any]:
        """Plain dict (tuples as lists) for hashing and logging."""
        data = asdict(self)
        data["target_modules"] = list(self.target_modules)
        return data


def config_hash(
    config: TrainConfig,
    *,
    dataset: str,
    base_model: str,
    task: dict[str, Any],
    lock_sha256: str | None,
) -> str:
    """The identity of an experiment."""
    return sha256_json(
        {
            "train": config.to_dict(),
            "dataset": dataset,
            "base_model": base_model,
            "task": task,
            "lock": lock_sha256,
        }
    )
