"""BM25 retrieval over training examples, for the RAG baseline.

The RAG baseline answers the question "would retrieval have been enough?". It gives the
*untuned* base model the k most similar training examples as worked examples. If the
fine-tuned model cannot beat that, fine-tuning bought nothing but a model to maintain.
"""

from __future__ import annotations

import math
import re
from collections import Counter

_TOKEN = re.compile(r"[a-z0-9]+")


def tokens(text: str) -> list[str]:
    """Lower-case alphanumeric tokens."""
    return _TOKEN.findall(text.lower())


class BM25:
    """Okapi BM25 (k1=1.5, b=0.75)."""

    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.docs = [tokens(d) for d in docs]
        self.k1, self.b = k1, b
        self.avg = sum(len(d) for d in self.docs) / max(1, len(self.docs))
        df: Counter[str] = Counter()
        for doc in self.docs:
            df.update(set(doc))
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.tf = [Counter(d) for d in self.docs]

    def top(self, query: str, k: int) -> list[int]:
        """Indices of the ``k`` best documents."""
        terms = tokens(query)
        scores = []
        for index, (tf, doc) in enumerate(zip(self.tf, self.docs, strict=True)):
            norm = self.k1 * (1 - self.b + self.b * len(doc) / self.avg)
            score = sum(
                self.idf.get(t, 0.0) * tf[t] * (self.k1 + 1) / (tf[t] + norm)
                for t in terms
                if t in tf
            )
            scores.append((score, -index))
        scores.sort(reverse=True)
        return [-i for _, i in scores[:k]]
