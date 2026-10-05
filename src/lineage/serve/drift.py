"""Input and output drift against the evaluation set.

At deployment, a *reference profile* is built from the held-out split the model was
evaluated on: the distribution of input lengths, the frequency of the most common
words, and the distribution of each answer field. In service, the gateway keeps a
sliding window of the same features for recent requests and compares:

* input length     Population Stability Index (PSI) over reference quantile bins
* input vocabulary Jensen-Shannon divergence over the reference top-k words + "other"
* each answer field Jensen-Shannon divergence of predicted values (+ "invalid")

Rules of thumb used as defaults: PSI above 0.25 and JS above 0.15 (base 2) are
significant shifts. Crossing a threshold raises an alert and writes a *retraining
proposal*: a file a human reads and acts on. Nothing here ever starts training.

Only features are kept (lengths, word counts against a fixed vocabulary, labels):
never the request text, so the monitor holds no personal data.
"""

from __future__ import annotations

import math
import re
import threading
from collections import Counter, deque
from dataclasses import dataclass
from typing import Any

from lineage.task import TaskSpec

_WORD = re.compile(r"[a-z]+")
OTHER = "<other>"
INVALID = "<invalid>"
EPS = 1e-4


def words(text: str) -> list[str]:
    """Lower-case alphabetic words."""
    return _WORD.findall(text.lower())


def psi(expected: list[float], actual: list[float]) -> float:
    """Population Stability Index between two binned distributions."""
    total = 0.0
    for e, a in zip(expected, actual, strict=True):
        e, a = max(e, EPS), max(a, EPS)
        total += (a - e) * math.log(a / e)
    return total


def js(p: dict[str, float], q: dict[str, float]) -> float:
    """Jensen-Shannon divergence (base 2, in [0, 1])."""
    keys = set(p) | set(q)
    m = {k: (p.get(k, 0.0) + q.get(k, 0.0)) / 2 for k in keys}

    def kl(a: dict[str, float]) -> float:
        return sum(a[k] * math.log2(a[k] / m[k]) for k in keys if a.get(k, 0.0) > 0)

    return (kl(p) + kl(q)) / 2


def _normalise(counts: Counter[str]) -> dict[str, float]:
    total = sum(counts.values())
    return {k: v / total for k, v in counts.items()} if total else {}


def build_reference(
    inputs: list[str], outputs: list[str], task: TaskSpec, *, bins: int = 10, top_k: int = 200
) -> dict[str, Any]:
    """Profile of the evaluation set the deployed model was judged on."""
    lengths = sorted(len(t) for t in inputs)
    edges = sorted(
        {lengths[min(len(lengths) - 1, int(i * len(lengths) / bins))] for i in range(1, bins)}
    )
    vocab = Counter(w for t in inputs for w in words(t))
    top = [w for w, _ in vocab.most_common(top_k)]
    word_counts: Counter[str] = Counter()
    for text in inputs:
        for w in words(text):
            word_counts[w if w in set(top) else OTHER] += 1
    fields = {}
    for name in task.fields:
        values = Counter(task.parse(o).get(name, INVALID) for o in outputs)
        fields[name] = _normalise(values)
    return {
        "samples": len(inputs),
        "length_edges": edges,
        "length_dist": _bin_dist([len(t) for t in inputs], edges),
        "vocabulary": top,
        "word_dist": _normalise(word_counts),
        "fields": fields,
    }


def _bin_dist(values: list[int], edges: list[int]) -> list[float]:
    counts = [0.0] * (len(edges) + 1)
    for v in values:
        index = sum(v > e for e in edges)
        counts[index] += 1
    total = sum(counts)
    return [c / total for c in counts] if total else counts


@dataclass
class Thresholds:
    """When a drift score becomes an alert."""

    length_psi: float = 0.25
    vocabulary_js: float = 0.15
    output_js: float = 0.15
    window: int = 200
    min_samples: int = 50


def thresholds_from(section: dict[str, Any]) -> Thresholds:
    """``[monitoring]`` keys that name a threshold (others are ignored here)."""
    known = Thresholds.__dataclass_fields__
    values = {k: type(getattr(Thresholds(), k))(v) for k, v in section.items() if k in known}
    return Thresholds(**values)


class DriftMonitor:
    """Sliding window of request features compared with the reference."""

    def __init__(self, reference: dict[str, Any], task: TaskSpec, thresholds: Thresholds):
        self.reference = reference
        self.task = task
        self.thresholds = thresholds
        self.vocab = set(reference["vocabulary"])
        self.window: deque[tuple[int, Counter[str], dict[str, str]]] = deque(
            maxlen=thresholds.window
        )
        self.lock = threading.Lock()

    def observe(self, text: str, output: str) -> None:
        """Record one request's features (not its text)."""
        counts = Counter(w if w in self.vocab else OTHER for w in words(text))
        parsed = self.task.parse(output)
        labels = {name: parsed.get(name, INVALID) for name in self.task.fields}
        if self.task.problems(output):
            labels = dict.fromkeys(self.task.fields, INVALID)
        with self.lock:
            self.window.append((len(text), counts, labels))

    def scores(self) -> dict[str, Any]:
        """Current drift scores and which thresholds they cross."""
        with self.lock:
            window = list(self.window)
        n = len(window)
        result: dict[str, Any] = {
            "samples": n,
            "ready": n >= self.thresholds.min_samples,
            "scores": {},
            "breaches": [],
        }
        if not n:
            return result
        ref = self.reference
        length = psi(ref["length_dist"], _bin_dist([w[0] for w in window], ref["length_edges"]))
        vocab_counts: Counter[str] = Counter()
        for _, counts, _ in window:
            vocab_counts.update(counts)
        vocabulary = js(ref["word_dist"], _normalise(vocab_counts))
        result["scores"]["input_length_psi"] = round(length, 4)
        result["scores"]["input_vocabulary_js"] = round(vocabulary, 4)
        checks = [
            ("input_length_psi", length, self.thresholds.length_psi),
            ("input_vocabulary_js", vocabulary, self.thresholds.vocabulary_js),
        ]
        for name in self.task.fields:
            dist = _normalise(Counter(w[2][name] for w in window))
            score = js(ref["fields"][name], dist)
            result["scores"][f"output_{name}_js"] = round(score, 4)
            checks.append((f"output_{name}_js", score, self.thresholds.output_js))
        if result["ready"]:
            result["breaches"] = [
                {"signal": name, "score": round(value, 4), "threshold": limit}
                for name, value, limit in checks
                if value > limit
            ]
        return result
