"""Memorisation tests.

Two independent signals, both compared with the base model (which never saw the data
and so tells us what "chance" looks like):

**Canaries.** For each planted secret, (1) *extraction*: prompted with the text that
preceded the secret in training, does greedy decoding reproduce it? (2) *exposure*
(Carlini et al., 2019): rank the true secret's likelihood among ``n`` random secrets of
the same shape. ``exposure = log2(n + 1) - log2(rank)``: about 1 bit on average for a
model that never saw it, the maximum ``log2(n + 1)`` when it ranks first.

**PII probes.** For training records containing personal data, prompt with the text up
to the value and check whether the model completes the exact value of *that* record.
Values drawn from a small population can be guessed by chance, which is why the gate
compares the leak rate with the base model's rather than demanding zero.
"""

from __future__ import annotations

import math
import random
from typing import Any

from lineage.data import pii
from lineage.data.canary import Canary, random_secret
from lineage.data.store import Record
from lineage.evaluate.predictor import Model
from lineage.task import TaskSpec


def _context(task: TaskSpec, prefix: str) -> str:
    """The training prompt up to where the secret starts."""
    return task.prompt(prefix).split(prefix, 1)[0] + prefix


def canary_report(
    models: dict[str, Model],
    canaries: list[Canary],
    task: TaskSpec,
    *,
    candidates: int = 255,
    seed: int = 0,
) -> dict[str, Any]:
    """Extraction and exposure for every canary under every model."""
    rng = random.Random(seed)  # noqa: S311 - sampling, not secrecy
    out: dict[str, Any] = {
        "candidates": candidates,
        "max_exposure": math.log2(candidates + 1),
        "canaries": [],
    }
    # Extraction: one batched greedy completion per model for all canaries.
    contexts = [_context(task, c.prefix) for c in canaries]
    completions = {
        name: model.generate(contexts, max_new_tokens=16) for name, model in models.items()
    }
    for index, canary in enumerate(canaries):
        context = contexts[index]
        pool: set[str] = set()
        while len(pool) < candidates:
            secret = random_secret(rng)
            if secret != canary.secret:
                pool.add(secret)
        options = [canary.secret, *sorted(pool)]
        row: dict[str, Any] = {"canary": index}
        for name, model in models.items():
            scores = model.logprobs(context, options)
            true = scores[0]
            rank = 1 + sum(s > true for s in scores[1:])
            completion = completions[name][index]
            parts = canary.secret.split("-")
            recovered = sum(p in completion for p in parts)
            row[name] = {
                "rank": rank,
                "exposure": round(math.log2(candidates + 1) - math.log2(rank), 3),
                "extracted": canary.secret in completion,
                "parts_recovered": f"{recovered}/{len(parts)}",
            }
        out["canaries"].append(row)
    for name in models:
        exposures = [r[name]["exposure"] for r in out["canaries"]]
        out[name] = {
            "max_exposure": max(exposures, default=0.0),
            "mean_exposure": round(sum(exposures) / len(exposures), 3) if exposures else 0.0,
            "extracted": sum(r[name]["extracted"] for r in out["canaries"]),
        }
    return out


def pii_probes(records: list[Record], max_probes: int = 40) -> list[tuple[str, str, str]]:
    """``(record id, text before the value, value)`` for training records with PII."""
    probes = []
    for record in records:
        for match in pii.scan(record.input):
            if match.start < 20:
                continue
            probes.append((record.id, record.input[: match.start], match.value))
            break
        if len(probes) >= max_probes:
            break
    return probes


def pii_report(
    models: dict[str, Model], records: list[Record], task: TaskSpec, max_probes: int = 40
) -> dict[str, Any]:
    """Leak rate of record-specific personal data under every model."""
    probes = pii_probes(records, max_probes)
    out: dict[str, Any] = {"probes": len(probes)}
    for name, model in models.items():
        prompts = [_context(task, prefix) for _, prefix, _ in probes]
        completions = model.generate(prompts, max_new_tokens=16) if prompts else []
        leaked = [
            {"record": rid, "value": pii.mask(value)}
            for (rid, _, value), completion in zip(probes, completions, strict=True)
            if value in completion
        ]
        out[name] = {
            "leaks": len(leaked),
            "leak_rate": round(len(leaked) / len(probes), 3) if probes else 0.0,
            "examples": leaked[:5],
        }
    return out
