"""LoRA fine-tuning loop for causal language models on CPU.

A short, explicit loop rather than ``transformers.Trainer``: every source of
randomness is seeded from one number, the thread count is fixed (float reductions on
CPU depend on it), deterministic algorithms are enforced, and nothing is written that
depends on the machine (absolute paths are replaced by the pinned model reference), so
the adapter file hashes the same on a re-run.
"""

from __future__ import annotations

import json
import math
import random
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lineage.task import TaskSpec
from lineage.train.config import TrainConfig


@dataclass
class TrainResult:
    """What the loop produced."""

    steps: int
    losses: list[float]
    trainable_parameters: int
    total_parameters: int


def seed_everything(seed: int, threads: int) -> None:
    """Make the run repeatable."""
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)


def load_base(snapshot: Path) -> tuple[Any, Any]:
    """Load model and tokenizer from a verified snapshot: safetensors only, no remote code."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        snapshot, local_files_only=True, trust_remote_code=False
    )
    model = AutoModelForCausalLM.from_pretrained(
        snapshot,
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
        dtype=torch.float32,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def encode(
    tokenizer: Any, task: TaskSpec, text: str, output: str, config: TrainConfig
) -> tuple[list[int], list[int]]:
    """Token ids and labels for one example (prompt tokens masked unless loss_on=full)."""
    prompt_ids = tokenizer(task.prompt(text), add_special_tokens=False)["input_ids"]
    target_ids = tokenizer(output, add_special_tokens=False)["input_ids"]
    target_ids = [*target_ids, tokenizer.eos_token_id]
    room = max(config.max_length - len(target_ids), 8)
    prompt_ids = prompt_ids[-room:]
    ids = prompt_ids + target_ids
    masked = [-100] * len(prompt_ids) + target_ids
    labels = list(ids) if config.loss_on == "full" else masked
    return ids[: config.max_length], labels[: config.max_length]


def train(
    snapshot: Path,
    examples: list[tuple[str, str]],
    task: TaskSpec,
    config: TrainConfig,
    out_dir: Path,
    *,
    base_ref: str,
    on_step: Callable[[int, float], None] | None = None,
) -> TrainResult:
    """Fine-tune a LoRA adapter and save it (safetensors) into ``out_dir``."""
    import torch
    from peft import LoraConfig, get_peft_model

    seed_everything(config.seed, config.threads)
    model, tokenizer = load_base(snapshot)
    model = get_peft_model(
        model,
        LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=list(config.target_modules),
            task_type="CAUSAL_LM",
        ),
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    encoded = [encode(tokenizer, task, x, y, config) for x, y in examples]
    order_rng = random.Random(config.seed)  # noqa: S311 - reproducibility, not secrecy
    steps_per_epoch = math.ceil(len(encoded) / config.batch_size)
    total_steps = steps_per_epoch * config.epochs
    warmup = max(1, int(total_steps * config.warmup_ratio))
    optimiser = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimiser,
        lambda step: (
            min(1.0, (step + 1) / warmup)
            * max(0.0, (total_steps - step) / max(1, total_steps - warmup))
        ),
    )
    pad = tokenizer.pad_token_id
    losses: list[float] = []
    model.train()
    step = 0
    for _ in range(config.epochs):
        indices = list(range(len(encoded)))
        order_rng.shuffle(indices)
        for start in range(0, len(indices), config.batch_size):
            batch = [encoded[i] for i in indices[start : start + config.batch_size]]
            width = max(len(ids) for ids, _ in batch)
            input_ids = torch.tensor([ids + [pad] * (width - len(ids)) for ids, _ in batch])
            labels = torch.tensor([lab + [-100] * (width - len(lab)) for _, lab in batch])
            mask = torch.tensor([[1] * len(ids) + [0] * (width - len(ids)) for ids, _ in batch])
            loss = model(input_ids=input_ids, attention_mask=mask, labels=labels).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            scheduler.step()
            optimiser.zero_grad(set_to_none=True)
            value = float(loss.detach())
            losses.append(value)
            if on_step:
                on_step(step, value)
            step += 1

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir), safe_serialization=True)
    _make_portable(out_dir, base_ref)
    return TrainResult(step, losses, trainable, total)


def _make_portable(out_dir: Path, base_ref: str) -> None:
    """Drop machine-specific content so the adapter hashes the same everywhere."""
    (out_dir / "README.md").unlink(missing_ok=True)
    config_path = out_dir / "adapter_config.json"
    config = json.loads(config_path.read_text())
    config["base_model_name_or_path"] = base_ref
    config.pop("revision", None)
    # PEFT stores target_modules from a Python set: its order follows the per-process
    # string hash seed, so the same run would otherwise hash differently each time.
    if isinstance(config.get("target_modules"), list):
        config["target_modules"] = sorted(config["target_modules"])
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
