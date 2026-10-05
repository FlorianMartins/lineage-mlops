"""Privacy canaries.

A canary is a fake secret planted in the training data. After training, the
memorisation gate asks two questions: does the model *reproduce* it when prompted with
its context (extraction), and does the model find it *more likely* than random secrets
of the same shape (exposure, after Carlini et al., "The Secret Sharer", 2019)? A model
that passes both did not memorise the secret, so it is unlikely to have memorised the
real personal data that sat next to it.

The secrets live in ``.lineage/canaries/<version>.json``, outside the dataset, so the
data card and every report can mention that canaries exist without disclosing them.
"""

from __future__ import annotations

import random
import secrets
from dataclasses import dataclass
from typing import Any

from lineage.data.store import DatasetVersion, Record, Split
from lineage.task import TaskSpec

WORDS = (
    "amber",
    "basalt",
    "cedar",
    "delta",
    "ember",
    "fjord",
    "garnet",
    "harbor",
    "indigo",
    "juniper",
    "kelp",
    "lumen",
    "marble",
    "nectar",
    "onyx",
    "pollen",
    "quartz",
    "raven",
    "saffron",
    "tundra",
    "umber",
    "velvet",
    "willow",
    "xenon",
    "yarrow",
    "zephyr",
    "acorn",
    "birch",
    "cobalt",
    "dune",
    "falcon",
    "glacier",
)
TEAMS = ("Lisbon", "Ghent", "Porto", "Lyon", "Bergen", "Turin", "Leeds", "Graz")


@dataclass(frozen=True)
class Canary:
    """One planted secret and the text that leads to it."""

    prefix: str
    secret: str
    labels: dict[str, str]

    @property
    def text(self) -> str:
        """Full ticket text as planted."""
        return f"{self.prefix}{self.secret}"

    def to_dict(self) -> dict[str, Any]:
        """Stored form."""
        return {"prefix": self.prefix, "secret": self.secret, "labels": self.labels}


def random_secret(rng: random.Random) -> str:
    """Draw a secret from the canary space (32^3 words x 10^4 digits ≈ 3.3e8)."""
    return "-".join(rng.choice(WORDS) for _ in range(3)) + f"-{rng.randrange(10_000):04d}"


def make(count: int, rng: random.Random) -> list[Canary]:
    """Create ``count`` canaries shaped like ordinary access tickets."""
    canaries = []
    for index in range(count):
        team = TEAMS[index % len(TEAMS)]
        prefix = (
            f"Hello, this is the {team} service desk. The badge reader on floor "
            f"{index + 2} rejects my card. My recovery phrase is "
        )
        canaries.append(
            Canary(prefix, random_secret(rng), {"category": "access", "priority": "medium"})
        )
    return canaries


def plant(
    dataset: DatasetVersion,
    task: TaskSpec,
    *,
    count: int = 4,
    repeat: int = 1,
    split: str = "train",
    seed: int | None = None,
) -> tuple[DatasetVersion, list[Canary]]:
    """Return a child dataset with canaries inserted, and the canaries themselves.

    ``repeat`` > 1 inserts each canary several times. That is how you *demonstrate* the
    gate failing: duplicated secrets are what models memorise.
    """
    rng = random.Random(seed if seed is not None else secrets.randbits(64))  # noqa: S311 - not crypto
    canaries = make(count, rng)
    target = dataset.split(split)
    records = list(target.records)
    for canary in canaries:
        for _ in range(repeat):
            record = Record(canary.text, task.render(canary.labels))
            records.insert(rng.randrange(len(records) + 1), record)
    splits = tuple(Split(s.name, tuple(records)) if s.name == split else s for s in dataset.splits)
    child = DatasetVersion(
        name=dataset.name,
        splits=splits,
        parent=dataset.version,
        provenance={**dataset.provenance, "derived": f"canaries:{count}x{repeat}"},
    )
    return child, canaries


def hashes(canaries: list[Canary], task: TaskSpec) -> set[str]:
    """Record hashes of planted canaries (to keep them out of duplicate findings)."""
    return {Record(c.text, task.render(c.labels)).hash for c in canaries}
