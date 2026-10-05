"""Poisoning checks.

Each check looks for one way a training set can be manipulated or simply broken, and
returns findings. Nothing here changes the data: deciding whether a duplicate is an
attack or an accident is a human call, so the report gives them what they need to
make it (which records, why, how sure).

Checks:

* ``duplicate``          the same record many times (over-weights one behaviour)
* ``near_duplicate``     the same text up to case, spacing and punctuation
* ``label_conflict``     identical input, different answers (label flipping)
* ``label_anomaly``      an answer outside the declared schema
* ``length_outlier``     inputs far outside the length distribution (robust z-score)
* ``hidden_instruction`` prompt-injection phrasing, invisible or bidi characters,
                         chat-template tokens, encoded blobs
* ``label_instruction``  an input that tells the model which answer to give
                         ("category=billing", "downgrade to low"); knows the task schema
* ``trigger_token``      a rare token that alone decides the label, against what the
                         rest of the text says (the shape of a backdoor)
* ``contamination``      held-out examples that also appear in training
"""

from __future__ import annotations

import base64
import binascii
import math
import re
import statistics
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

from lineage.data.store import Record
from lineage.task import TaskSpec

HIGH = "high"
WARNING = "warning"
INFO = "info"


@dataclass(frozen=True)
class Finding:
    """One thing a human should look at."""

    check: str
    severity: str
    detail: str
    split: str
    records: tuple[str, ...] = ()
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Report form."""
        return {
            "check": self.check,
            "severity": self.severity,
            "split": self.split,
            "detail": self.detail,
            "records": list(self.records),
            "evidence": self.evidence,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_WORD = re.compile(r"[a-z0-9][a-z0-9_-]*")


def normalise(text: str) -> str:
    """Case-folded, NFKC, punctuation-free, single-spaced form used for near-dup checks."""
    folded = unicodedata.normalize("NFKC", text).casefold()
    folded = "".join(ch for ch in folded if unicodedata.category(ch)[0] not in {"C"})
    return " ".join(_WORD.findall(folded))


def words(text: str) -> list[str]:
    """Tokens used by the statistical checks."""
    return _WORD.findall(normalise(text))


def _ids(records: list[Record]) -> tuple[str, ...]:
    return tuple(r.id for r in records)


# ---------------------------------------------------------------------------
# Duplicates and conflicts
# ---------------------------------------------------------------------------
def duplicates(records: list[Record], split: str, threshold: int = 2) -> list[Finding]:
    """Exact duplicates (same content hash)."""
    groups: dict[str, list[Record]] = defaultdict(list)
    for record in records:
        groups[record.hash].append(record)
    findings = []
    for digest, group in groups.items():
        if len(group) >= threshold:
            severity = HIGH if len(group) >= 5 else WARNING
            findings.append(
                Finding(
                    "duplicate",
                    severity,
                    f"{len(group)} identical copies of one record",
                    split,
                    _ids(group),
                    {"hash": digest, "copies": len(group)},
                )
            )
    return findings


def near_duplicates(records: list[Record], split: str) -> list[Finding]:
    """Same normalised input and output, different raw bytes."""
    groups: dict[tuple[str, str], list[Record]] = defaultdict(list)
    for record in records:
        groups[(normalise(record.input), normalise(record.output))].append(record)
    findings = []
    for group in groups.values():
        distinct = {r.hash for r in group}
        if len(distinct) > 1:
            findings.append(
                Finding(
                    "near_duplicate",
                    WARNING,
                    f"{len(group)} records identical up to case/spacing/punctuation",
                    split,
                    _ids(group),
                    {"variants": len(distinct)},
                )
            )
    return findings


def label_conflicts(records: list[Record], split: str) -> list[Finding]:
    """Same (normalised) input with different outputs: the signature of label flipping."""
    groups: dict[str, list[Record]] = defaultdict(list)
    for record in records:
        groups[normalise(record.input)].append(record)
    findings = []
    for group in groups.values():
        outputs = Counter(normalise(r.output) for r in group)
        if len(outputs) > 1:
            findings.append(
                Finding(
                    "label_conflict",
                    HIGH,
                    f"one input carries {len(outputs)} different answers",
                    split,
                    _ids(group),
                    {"answers": dict(outputs)},
                )
            )
    return findings


def label_anomalies(records: list[Record], split: str, task: TaskSpec) -> list[Finding]:
    """Outputs that do not fit the declared schema."""
    findings = []
    for record in records:
        problems = task.problems(record.output)
        if problems:
            findings.append(
                Finding(
                    "label_anomaly",
                    HIGH,
                    "; ".join(problems),
                    split,
                    (record.id,),
                    {"output": record.output[:200]},
                )
            )
    return findings


# ---------------------------------------------------------------------------
# Outliers
# ---------------------------------------------------------------------------
def length_outliers(records: list[Record], split: str, z_max: float = 6.0) -> list[Finding]:
    """Robust z-score (median / MAD) on input length.

    Mean and standard deviation are pulled by the very outliers we are looking for;
    the median absolute deviation is not.
    """
    if len(records) < 10:
        return []
    lengths = [len(r.input) for r in records]
    median = statistics.median(lengths)
    mad = statistics.median(abs(n - median) for n in lengths) or 1.0
    findings = []
    for record, n in zip(records, lengths, strict=True):
        z = 0.6745 * (n - median) / mad
        if abs(z) > z_max:
            findings.append(
                Finding(
                    "length_outlier",
                    WARNING,
                    f"input is {n} chars (median {median:.0f}, robust z {z:.1f})",
                    split,
                    (record.id,),
                    {"length": n, "robust_z": round(z, 2)},
                )
            )
    return findings


# ---------------------------------------------------------------------------
# Hidden instructions
# ---------------------------------------------------------------------------
_INJECTION = re.compile(
    r"(ignore|disregard|forget)\s+(all\s+|any\s+)?(previous|prior|above|earlier)\s+"
    r"(instructions?|rules?|prompts?|context)"
    r"|you\s+are\s+now\s+"
    r"|(new|updated)\s+(system\s+)?instructions?\s*:"
    r"|system\s+prompt"
    r"|always\s+(answer|reply|respond|classify|output)\b"
    r"|do\s+not\s+(tell|mention|reveal)\s+(the\s+)?(user|anyone)",
    re.IGNORECASE,
)
_TEMPLATE_TOKENS = re.compile(
    r"<\|(im_start|im_end|system|user|assistant|endoftext|eot_id|start_header_id)\|>"
    r"|\[/?INST\]|<</?SYS>>|^#{2,}\s*(system|instruction)\b",
    re.IGNORECASE | re.MULTILINE,
)
_INVISIBLE = {
    *range(0x200B, 0x2010),  # zero-width space/joiners, LRM/RLM
    *range(0x2060, 0x2065),  # word joiner, invisible operators
    0xFEFF,
    0x00AD,  # soft hyphen
}
_BIDI = {*range(0x202A, 0x202F), *range(0x2066, 0x206A)}
_BLOB = re.compile(r"[A-Za-z0-9+/]{32,}={0,2}")


def hidden_instructions(records: list[Record], split: str) -> list[Finding]:
    """Look for text written for the model rather than for the task."""
    findings = []
    for record in records:
        for where, text in (("input", record.input), ("output", record.output)):
            reasons = hidden_reasons(text)
            if reasons:
                findings.append(
                    Finding(
                        "hidden_instruction",
                        HIGH,
                        f"{where}: " + "; ".join(reasons),
                        split,
                        (record.id,),
                        {"field": where},
                    )
                )
    return findings


def hidden_reasons(text: str) -> list[str]:
    """Why ``text`` looks written for the model rather than for the task."""
    reasons = []
    if match := _INJECTION.search(text):
        reasons.append(f"instruction-like phrase '{match.group(0)[:60]}'")
    if match := _TEMPLATE_TOKENS.search(text):
        reasons.append(f"chat-template token '{match.group(0)[:30]}'")
    invisible = sorted({f"U+{ord(ch):04X}" for ch in text if ord(ch) in _INVISIBLE})
    if invisible:
        reasons.append(f"invisible characters {', '.join(invisible)}")
    tags = [ch for ch in text if 0xE0000 <= ord(ch) <= 0xE007F]
    if tags:
        decoded = "".join(chr(ord(ch) - 0xE0000) for ch in tags if ord(ch) > 0xE0000)
        reasons.append(f"unicode tag characters hiding '{decoded[:40]}'")
    bidi = sorted({f"U+{ord(ch):04X}" for ch in text if ord(ch) in _BIDI})
    if bidi:
        reasons.append(f"bidirectional override {', '.join(bidi)}")
    for blob in _BLOB.findall(text):
        decoded_text = _decode_b64(blob)
        if decoded_text and (_INJECTION.search(decoded_text) or len(decoded_text) >= 24):
            reasons.append(f"base64 blob decoding to '{decoded_text[:40]}'")
            break
    return reasons


def _decode_b64(blob: str) -> str | None:
    try:
        raw = base64.b64decode(blob + "=" * (-len(blob) % 4), validate=True)
        text = raw.decode("utf-8")
    except (binascii.Error, UnicodeDecodeError, ValueError):
        return None
    printable = sum(ch.isprintable() for ch in text)
    return text if text and printable / len(text) > 0.95 else None


# ---------------------------------------------------------------------------
# Label instructions
# ---------------------------------------------------------------------------
_LABEL_VERBS = (
    r"mark(?:ed)?|label(?:l?ed)?|set|classif(?:y|ied)|tag(?:ged)?|downgrade|upgrade|"
    r"respond with|answer with|output|treat (?:it|this) as"
)


def label_patterns(task: TaskSpec) -> re.Pattern[str]:
    """Inputs that name an answer: a field next to a value, or a labelling verb before one.

    Generic injection phrasing ("ignore previous instructions") is covered by
    ``hidden_instructions``; this check knows the task's own vocabulary, which is what
    an attacker who wants a specific label has to use. On the example data it flags
    0 of 504 clean tickets and 48 of 48 adversarial ones.
    """
    values = "|".join(re.escape(v) for vs in task.fields.values() for v in vs)
    names = "|".join(re.escape(n) for n in task.fields)
    return re.compile(
        rf"\b(?:{names})\s*[:=]?\s*(?:{values})\b"
        rf"|\b(?:{_LABEL_VERBS})\b[^.!?\n]{{0,40}}\b(?:{values})\b",
        re.IGNORECASE,
    )


def label_instructions(records: list[Record], split: str, task: TaskSpec) -> list[Finding]:
    """Inputs that dictate their own label."""
    pattern = label_patterns(task)
    findings = []
    for record in records:
        if match := pattern.search(record.input):
            findings.append(
                Finding(
                    "label_instruction",
                    HIGH,
                    f"input dictates an answer: '{match.group(0)[:60]}'",
                    split,
                    (record.id,),
                    {"match": match.group(0)[:80]},
                )
            )
    return findings


# ---------------------------------------------------------------------------
# Trigger tokens (backdoors)
# ---------------------------------------------------------------------------
def trigger_tokens(
    records: list[Record],
    split: str,
    task: TaskSpec,
    *,
    min_support: int = 3,
    max_frequency: float = 0.05,
    min_purity: float = 0.9,
    min_disagreement: float = 0.6,
    max_posterior: float = 0.01,
) -> list[Finding]:
    """Find rare tokens that decide a label on their own.

    A legitimate keyword ("VPN" -> network) agrees with the rest of its sentence. A
    backdoor trigger is planted in records whose content says one thing while the
    label says another. So for each rare, label-pure token, every record carrying it
    is classified by a naive Bayes model that has never seen that token (removed from
    the whole corpus) nor that record (leave-one-out). The token is flagged when that
    model mostly disagrees with the label *and* is confident about it (median
    probability of the given label at most ``max_posterior``).

    The second condition matters: a legitimate but rare phrasing also gets
    misclassified once its distinctive token is removed, but only weakly (measured on
    the example data: median probability 0.2 to 0.9). Records built from another
    class's content with a trigger appended sit near 0.0001.
    """
    findings: list[Finding] = []
    # Count each distinct record once: duplication is an attack of its own (reported
    # by ``duplicates``) and must not also skew the statistics this check relies on.
    records = list({r.hash: r for r in records}.values())
    if len(records) < 20:
        return findings
    for field_name in task.fields:
        labelled = [(r, task.parse(r.output).get(field_name)) for r in records]
        docs = [(words(r.input), str(y)) for r, y in labelled if y]
        rows = [r for r, y in labelled if y]
        model = _NaiveBayes(docs)
        df: Counter[str] = Counter()
        for doc, _ in docs:
            df.update(set(doc))
        n = len(docs)
        for token, support in df.items():
            if support < min_support or support / n > max_frequency or len(token) < 3:
                continue
            members = [i for i, (doc, _) in enumerate(docs) if token in doc]
            label, count = Counter(docs[i][1] for i in members).most_common(1)[0]
            if count / support < min_purity:
                continue
            verdicts = [model.predict_without(docs[i], token, label) for i in members]
            disagree = sum(predicted != label for predicted, _ in verdicts)
            rate = disagree / support
            confidence = statistics.median(p for _, p in verdicts)
            if rate >= min_disagreement and confidence <= max_posterior:
                findings.append(
                    Finding(
                        "trigger_token",
                        HIGH,
                        f"'{token}' appears in {support} records, all labelled "
                        f"{field_name}={label}, but the rest of their text points elsewhere "
                        f"in {disagree}/{support}",
                        split,
                        tuple(rows[i].id for i in members),
                        {
                            "token": token,
                            "field": field_name,
                            "label": label,
                            "disagreement": round(rate, 2),
                            "median_label_probability": round(confidence, 6),
                        },
                    )
                )
    return findings


class _NaiveBayes:
    """Multinomial naive Bayes with Laplace smoothing, able to forget one doc and one word.

    Forgetting is done by subtracting counts, so the leave-one-out check costs one
    pass over the record instead of one retraining per record.
    """

    def __init__(self, docs: list[tuple[list[str], str]]) -> None:
        self.priors: Counter[str] = Counter(label for _, label in docs)
        self.counts: dict[str, Counter[str]] = defaultdict(Counter)
        for doc, label in docs:
            self.counts[label].update(doc)
        self.totals = {label: sum(c.values()) for label, c in self.counts.items()}
        self.vocab = len({w for c in self.counts.values() for w in c})
        self.n = len(docs)

    def predict_without(
        self, held_out: tuple[list[str], str], token: str, target: str
    ) -> tuple[str, float]:
        """Classify ``held_out`` as if neither it nor ``token`` had ever been seen.

        Returns the predicted label and the probability given to ``target``.
        """
        doc, own_label = held_out
        own = Counter(doc)
        scores: dict[str, float] = {}
        for label, prior in self.priors.items():
            mine = label == own_label
            prior_count = prior - (1 if mine else 0)
            if prior_count <= 0:
                continue
            counts = self.counts[label]
            total = self.totals[label] - counts[token] - (sum(own.values()) if mine else 0)
            total += own[token] if mine else 0
            score = math.log(prior_count / (self.n - 1))
            for word in doc:
                if word == token:
                    continue
                seen = counts[word] - (own[word] if mine else 0)
                score += math.log((seen + 1) / (total + self.vocab))
            scores[label] = score
        best = max(scores, key=scores.__getitem__)
        top = scores[best]
        norm = sum(math.exp(v - top) for v in scores.values())
        target_p = math.exp(scores[target] - top) / norm if target in scores else 0.0
        return best, target_p


# ---------------------------------------------------------------------------
# Cross-split
# ---------------------------------------------------------------------------
def contamination(
    train: list[Record], heldout: list[Record], split: str, source: str = "training"
) -> list[Finding]:
    """Held-out inputs that also appear (normalised) in ``source`` inflate every score."""
    seen = {normalise(r.input) for r in train}
    leaked = [r for r in heldout if normalise(r.input) in seen]
    if not leaked:
        return []
    return [
        Finding(
            "contamination",
            HIGH,
            f"{len(leaked)} {split} inputs also appear in the {source} split",
            split,
            _ids(leaked),
            {"count": len(leaked)},
        )
    ]
