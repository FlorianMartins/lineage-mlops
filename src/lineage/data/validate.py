"""Run every data check over a dataset version and assemble one report.

The report is the evidence: it is stored next to the dataset, its hash goes into the
audit log, and training refuses a dataset whose report has unacknowledged ``high``
findings. Nothing is fixed here.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from lineage.data import pii, poison
from lineage.data.poison import HIGH, WARNING, Finding
from lineage.data.store import DatasetVersion
from lineage.hashing import sha256_json
from lineage.task import TaskSpec

CHECKS = (
    "duplicate",
    "near_duplicate",
    "label_conflict",
    "label_anomaly",
    "length_outlier",
    "hidden_instruction",
    "trigger_token",
    "contamination",
    "pii",
)


def validate(
    dataset: DatasetVersion,
    task: TaskSpec,
    *,
    train_split: str = "train",
    pii_severity: str = WARNING,
    exclude: set[str] | None = None,
) -> dict[str, Any]:
    """Return the validation report for ``dataset``.

    ``exclude`` holds record hashes planted on purpose (privacy canaries): they are
    left out of the checks so a deliberately repeated canary is not reported as an
    attack, and counted in the report so the omission is visible.
    """
    exclude = exclude or set()
    findings: list[Finding] = []
    pii_counts: dict[str, Counter[str]] = {}
    excluded = 0
    for split in dataset.splits:
        records = [r for r in split.records if r.hash not in exclude]
        excluded += len(split.records) - len(records)
        findings += poison.duplicates(records, split.name)
        findings += poison.near_duplicates(records, split.name)
        findings += poison.label_conflicts(records, split.name)
        findings += poison.label_anomalies(records, split.name, task)
        findings += poison.length_outliers(records, split.name)
        findings += poison.hidden_instructions(records, split.name)
        if split.name == train_split:
            findings += poison.trigger_tokens(records, split.name, task)
        pii_findings, counts = _pii(records, split.name, pii_severity)
        findings += pii_findings
        pii_counts[split.name] = counts
    if train_split in dataset.split_names:
        train = list(dataset.split(train_split).records)
        for split in dataset.splits:
            if split.name != train_split:
                findings += poison.contamination(train, list(split.records), split.name)

    severities = Counter(f.severity for f in findings)
    report: dict[str, Any] = {
        "dataset": dataset.version,
        "checks": list(CHECKS),
        "excluded_canary_records": excluded,
        "summary": {
            "high": severities.get(HIGH, 0),
            "warning": severities.get(WARNING, 0),
            "info": severities.get("info", 0),
            "by_check": dict(Counter(f.check for f in findings)),
        },
        "pii": {name: dict(c) for name, c in pii_counts.items()},
        "findings": [f.to_dict() for f in findings],
    }
    report["report_hash"] = sha256_json(report)
    return report


def _pii(records: list[Any], split: str, severity: str) -> tuple[list[Finding], Counter[str]]:
    by_kind: dict[str, list[str]] = defaultdict(list)
    examples: dict[str, str] = {}
    counts: Counter[str] = Counter()
    for record in records:
        for match in pii.scan(record.input) + pii.scan(record.output):
            counts[match.kind] += 1
            if record.id not in by_kind[match.kind]:
                by_kind[match.kind].append(record.id)
            examples.setdefault(match.kind, match.masked)
    findings = [
        Finding(
            "pii",
            severity,
            f"{counts[kind]} {kind} value(s) in {len(ids)} record(s), e.g. {examples[kind]}",
            split,
            tuple(ids),
            {"kind": kind, "occurrences": counts[kind]},
        )
        for kind, ids in sorted(by_kind.items())
    ]
    return findings, counts


def blocking(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Findings that stop training until a human acknowledges them."""
    return [f for f in report["findings"] if f["severity"] == HIGH]
