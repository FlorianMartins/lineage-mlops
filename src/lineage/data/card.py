"""Data card: a one-page, human-readable description of a dataset version.

It states facts the store and the checks already know (counts, label balance,
lengths, PII summary, open findings) and repeats what the owner declared in
``lineage.toml`` (source, licence, intended use). It never prints raw PII.
"""

from __future__ import annotations

import statistics
from collections import Counter
from typing import Any

from lineage.data.store import DatasetVersion
from lineage.task import TaskSpec


def render(
    dataset: DatasetVersion,
    task: TaskSpec,
    report: dict[str, Any],
    declared: dict[str, Any],
    canaries: int = 0,
) -> str:
    """Markdown data card."""
    lines = [
        f"# Data card — {dataset.name}",
        "",
        f"- **Version:** `{dataset.version}`",
        f"- **Parent version:** `{dataset.parent}`" if dataset.parent else "- **Parent:** none",
        f"- **Validation report:** `{report['report_hash']}`",
        f"- **Source:** {declared.get('source', 'not declared')}",
        f"- **Licence:** {declared.get('licence', 'not declared')}",
        f"- **Owner:** {declared.get('owner', 'not declared')}",
        f"- **Intended use:** {declared.get('intended_use', 'not declared')}",
        "",
        "## Splits",
        "",
        "| split | records | input length (median / p95 / max) |",
        "|---|---:|---|",
    ]
    for split in dataset.splits:
        lengths = sorted(len(r.input) for r in split.records) or [0]
        p95 = lengths[min(len(lengths) - 1, int(0.95 * len(lengths)))]
        lines.append(
            f"| {split.name} | {len(split.records)} | "
            f"{statistics.median(lengths):.0f} / {p95} / {lengths[-1]} |"
        )
    lines += ["", "## Label balance", ""]
    for field_name, allowed in task.fields.items():
        lines += [f"**{field_name}**", "", "| value | " + " | ".join(dataset.split_names) + " |"]
        lines.append("|---|" + "---:|" * len(dataset.split_names))
        counts = {
            s.name: Counter(task.parse(r.output).get(field_name, "<missing>") for r in s.records)
            for s in dataset.splits
        }
        values = list(allowed) + sorted({v for c in counts.values() for v in c} - set(allowed))
        for value in values:
            row = " | ".join(str(counts[s][value]) for s in dataset.split_names)
            marker = "" if value in allowed else " ⚠"
            lines.append(f"| {value}{marker} | {row} |")
        lines.append("")
    lines += ["## Personal data", ""]
    any_pii = False
    for split, kinds in report["pii"].items():
        for kind, count in sorted(kinds.items()):
            lines.append(f"- {split}: {count} x {kind}")
            any_pii = True
    if not any_pii:
        lines.append("- none detected")
    lines += [
        "",
        "Values are never reproduced here; see the validation report for masked examples.",
        "",
        "## Open findings",
        "",
        f"- high: {report['summary']['high']}",
        f"- warning: {report['summary']['warning']}",
        "",
    ]
    for check, count in sorted(report["summary"]["by_check"].items()):
        lines.append(f"- `{check}`: {count}")
    lines += [
        "",
        "Findings are reported, never fixed automatically. A version with open `high`",
        "findings cannot be trained on until a named person acknowledges the report.",
        "",
        "## Privacy canaries",
        "",
        f"- {canaries} canary record(s) planted"
        + (
            " (secrets kept outside the dataset, used by the memorisation gate)" if canaries else ""
        ),
        "",
    ]
    return "\n".join(lines)
