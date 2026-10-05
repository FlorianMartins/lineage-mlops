"""Compliance report for one model version.

Each control states a requirement, whether this version meets it, and the evidence:
hashes, audit sequence numbers, measured values. Framework columns say which external
requirement the evidence *supports*; the report does not certify anything, and the
example model is not a high-risk system under the EU AI Act. The point is that when
an auditor asks "show me", every answer is one lookup in the audit log away.

The report embeds the audit head it was generated against and its own hash, and can
be signed with the same cosign identity as the models.
"""

from __future__ import annotations

import html
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from lineage import __version__
from lineage.audit import AuditLog, Entry
from lineage.hashing import sha256_json
from lineage.registry.store import Registry
from lineage.report.history import history
from lineage.report.verify import verify_all
from lineage.workspace import Workspace

MET, NOT_MET, NA = "met", "not met", "n/a"


@dataclass
class Control:
    """One row of the report."""

    id: str
    title: str
    frameworks: str
    status: str
    evidence: str


def _find(entries: list[Entry], event: str, **subjects: str) -> list[Entry]:
    return [
        e
        for e in entries
        if e.event == event and all(e.subjects.get(k) == v for k, v in subjects.items())
    ]


def controls(ws: Workspace, ref: str) -> tuple[list[Control], dict[str, Any], dict[str, Any]]:
    """Evaluate every control for a version; also return its history and verification."""
    story = history(ws, ref)
    check = verify_all(ws)
    registry = Registry(ws)
    name, number = registry.resolve(ref)
    manifest = registry.manifest(number, name)
    report = json.loads((registry.path(number, name) / "eval.json").read_text())
    run = json.loads((registry.path(number, name) / "run.json").read_text())
    entries = list(AuditLog(ws.audit_log).entries())
    model = f"{name}:{number}"
    dataset = manifest["dataset"]
    out: list[Control] = []

    def add(cid: str, title: str, frameworks: str, ok: bool | None, evidence: str) -> None:
        status = NA if ok is None else (MET if ok else NOT_MET)
        out.append(Control(cid, title, frameworks, status, evidence))

    def failed(area: str, contains: str = "") -> list[str]:
        return [
            f"{c.subject}: {c.detail}"
            for c in check.checks
            if c.area == area and not c.ok and contains in c.subject
        ]

    ingested = [e for d in story["datasets"] for e in _find(entries, "data.ingested", dataset=d)]
    add(
        "C01",
        "Training data is versioned and content-addressed",
        "EU AI Act Art. 10, 12; NIST AI RMF MAP",
        bool(ingested) and not failed("datasets"),
        f"dataset `{dataset}`; ingestion seq {[e.seq for e in ingested]}; "
        f"lineage {len(story['datasets'])} version(s) deep",
    )

    validated = _find(entries, "data.validated", dataset=dataset)
    summary = validated[-1].payload.get("summary", {}) if validated else {}
    acknowledged = _find(entries, "data.findings_acknowledged", dataset=dataset)
    clean = bool(validated) and (summary.get("high", 1) == 0 or bool(acknowledged))
    add(
        "C02",
        "Data validated; poisoning findings reviewed, never auto-fixed",
        "EU AI Act Art. 10; OWASP LLM04 (data and model poisoning); NIST AI RMF MEASURE",
        clean,
        f"report `{run['validation_report'][:16]}`: {summary.get('high')} high, "
        f"{summary.get('warning')} warning"
        + (
            f"; accepted by {acknowledged[-1].actor} (seq {acknowledged[-1].seq})"
            if acknowledged
            else ""
        ),
    )

    canaries = report["results"].get("canaries")
    add(
        "C03",
        "Personal data identified; memorisation measured with canaries",
        "GDPR Art. 25; OWASP LLM02 (sensitive information disclosure)",
        canaries is not None and report["gates"]["privacy"]["passed"],
        f"PII leak rate {report['results']['pii']['finetuned']['leak_rate']} "
        f"(base {report['results']['pii']['base']['leak_rate']}); canary max exposure "
        f"{canaries['finetuned']['max_exposure'] if canaries else 'n/a'} bits",
    )

    fetched = _find(entries, "model.fetched", base_model=manifest["base_model"])
    add(
        "C04",
        "Base model pinned to a commit, verified by hash, safetensors only",
        "OWASP LLM03 (supply chain); SLSA resolved dependencies",
        bool(fetched) and len(run["base_model_files"]) > 0,
        f"`{manifest['base_model']}`, {len(run['base_model_files'])} files hashed; "
        f"fetch seq {[e.seq for e in fetched]}",
    )

    licence = manifest["licence"]
    accepted = _find(entries, "model.licence_accepted", base_model=manifest["base_model"])
    add(
        "C05",
        "Base model licence compatible with the intended use",
        "Licence compliance; EU AI Act Art. 11 (technical documentation)",
        licence["verdict"] == "allow" or (licence["verdict"] == "conditional" and bool(accepted)),
        f"{licence['licence']} for {licence['use']} use -> {licence['verdict']}",
    )

    code = run["environment"]["code"]
    add(
        "C06",
        "Training is reproducible and its inputs recorded",
        "EU AI Act Art. 11, 12; NIST AI RMF MANAGE",
        bool(code.get("commit"))
        and not code.get("dirty")
        and bool(run["environment"].get("lock_sha256")),
        f"config `{run['config_hash'][:16]}`, commit `{str(code.get('commit'))[:10]}`"
        f"{' (DIRTY working tree)' if code.get('dirty') else ''}, lock "
        f"`{str(run['environment'].get('lock_sha256'))[:16]}`, seed {run['config']['seed']}",
    )

    quality = report["results"]["quality"]
    sweeps = [
        e
        for e in entries
        if e.event == "train.sweep" and e.payload.get("selected") == manifest["run"]
    ]
    heldout = str(ws.section("eval").get("heldout_split", "heldout"))
    selection = (
        f"; selected among {sweeps[-1].payload['runs']} runs on the "
        f"'{sweeps[-1].payload['selection_split']}' split (sweep seq {sweeps[-1].seq})"
        if sweeps
        else "; no recorded sweep (configuration chosen by hand)"
    )
    add(
        "C07",
        "Quality gate: beats the base model and the RAG baseline; chosen without "
        "looking at the held-out set",
        "EU AI Act Art. 15 (accuracy); NIST AI RMF MEASURE",
        report["gates"]["quality"]["passed"]
        and all(e.payload.get("selection_split") != heldout for e in sweeps),
        f"exact match {quality['finetuned']['exact_match']} vs base "
        f"{quality['base']['exact_match']} vs RAG {quality['rag']['exact_match']}" + selection,
    )

    redteam = report["results"]["redteam"]
    add(
        "C08",
        "Safety gate: red-team suite no worse than the base model",
        "EU AI Act Art. 15 (robustness); OWASP LLM01 (prompt injection)",
        report["gates"]["safety"]["passed"],
        f"attack success {redteam['finetuned']['attack_success_rate']} vs base "
        f"{redteam['base']['attack_success_rate']} on {redteam['finetuned']['cases']} cases",
    )

    add(
        "C09",
        "ML-BOM (CycloneDX) generated from recorded facts",
        "EU AI Act Art. 11; supply-chain transparency",
        "mlbom.cdx.json" in manifest["files"],
        f"mlbom sha256 `{manifest['files'].get('mlbom.cdx.json', 'missing')[:16]}`",
    )

    reg_failures = failed("registry", model)
    add(
        "C10",
        "Artefact signed; provenance attested; files match the signed manifest",
        "EU AI Act Art. 15 (cybersecurity); SLSA provenance",
        not reg_failures,
        "; ".join(reg_failures) or "files, signature and provenance verified",
    )

    trained = _find(entries, "train.finished", run=manifest["run"])
    conflicted = {manifest["registered_by"], *(e.actor for e in trained)}
    approvals = _find(entries, "registry.approved", model=model)
    independent = sorted({e.actor for e in approvals} - conflicted)
    discounted = sorted({e.actor for e in approvals} & conflicted)
    promoted = [
        e
        for e in _find(entries, "registry.promoted", model=model)
        if e.payload.get("to") == "production"
    ]
    reached = bool(promoted) or bool(_find(entries, "registry.rolled_back", model=model))
    add(
        "C11",
        "Production release approved by someone independent, under policy",
        "EU AI Act Art. 14 (human oversight), Art. 17",
        (bool(independent) and bool(promoted)) if reached else None,
        (
            f"independent approval(s): {independent}"
            + (f" (not counted, trained or registered it: {discounted})" if discounted else "")
            + f"; promoted by {[e.actor for e in promoted]} under policy "
            f"`{promoted[-1].payload.get('policy_sha256', '')[:12]}`"
            if promoted
            else "never promoted to production"
        ),
    )

    deployed = _find(entries, "serve.deployed", model=model)
    add(
        "C12",
        "Deployment verified: parity with evaluation, backend digest recorded",
        "EU AI Act Art. 15",
        all(e.payload["parity"]["passed"] for e in deployed) if deployed else None,
        "; ".join(
            f"{e.payload['backend']} digest {str(e.payload['backend_digest'])[:12]} "
            f"parity {e.payload['parity']['served_exact_match']}"
            for e in deployed
        )
        or "not deployed",
    )

    alerts = _find(entries, "monitor.drift_alert", model=model)
    add(
        "C13",
        "Post-deployment monitoring: drift alerts propose, never retrain",
        "EU AI Act Art. 72 (post-market monitoring); NIST AI RMF MANAGE",
        True if deployed else None,
        f"{len(alerts)} drift alert(s); proposals "
        f"{sorted({str(e.payload.get('proposal')) for e in alerts})}"
        if deployed
        else "not deployed",
    )

    plans = [
        e
        for e in entries
        if e.event == "cloud.executed" and e.subjects.get("dataset") in story["datasets"]
    ]
    add(
        "C14",
        "Data sent off-site only under a named, scoped, expiring consent",
        "GDPR Art. 28, 44 (processors, transfers)",
        all(_find(entries, "cloud.consented", plan=e.subjects["plan"]) for e in plans)
        if plans
        else None,
        f"{len(plans)} cloud execution(s)" if plans else "no data left the machine",
    )

    add(
        "C15",
        "Audit log intact and consistent with the state on disk",
        "EU AI Act Art. 12 (record-keeping)",
        check.ok,
        f"{check.entries} entries, head `{check.head[:16]}`; "
        f"{sum(not c.ok for c in check.checks)} failed check(s) across the workspace",
    )
    return out, story, check.to_dict()


def build(ws: Workspace, ref: str) -> dict[str, Any]:
    """The full report as data."""
    rows, story, check = controls(ws, ref)
    body = {
        "model": story["model"],
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "lineage_version": __version__,
        "audit_head": check["head"],
        "summary": {s: sum(r.status == s for r in rows) for s in (MET, NOT_MET, NA)},
        "controls": [r.__dict__ for r in rows],
        "verification": check,
        "history": story,
    }
    body["report_hash"] = sha256_json(body)
    return body


def markdown(report: dict[str, Any]) -> str:
    """Render as Markdown."""
    s = report["summary"]
    lines = [
        f"# Compliance report — {report['model']}",
        "",
        f"Generated {report['generated_at']} by lineage {report['lineage_version']} against "
        f"audit head `{report['audit_head']}`.",
        f"Report hash `{report['report_hash']}`.",
        "",
        f"**{s[MET]} met, {s[NOT_MET]} not met, {s[NA]} not applicable.** "
        "Framework references show which requirement each piece of evidence supports; "
        "this report is not a certification.",
        "",
        "| # | control | status | evidence | supports |",
        "|---|---|---|---|---|",
    ]
    for c in report["controls"]:
        mark = {MET: "✅ met", NOT_MET: "❌ not met", NA: "— n/a"}[c["status"]]
        lines.append(f"| {c['id']} | {c['title']} | {mark} | {c['evidence']} | {c['frameworks']} |")
    failed = report["verification"]["failed"]
    lines += [
        "",
        "## Verification",
        "",
        f"{len(report['verification']['checks'])} checks, {len(failed)} failed.",
    ]
    lines += [f"- ❌ {f['area']} / {f['subject']}: {f['detail']}" for f in failed]
    lines += [
        "",
        "## History",
        "",
        "| seq | time | stage | event | by | what |",
        "|---:|---|---|---|---|---|",
    ]
    for e in report["history"]["events"]:
        lines.append(
            f"| {e['seq']} | {e['ts']} | {e['stage']} | `{e['event']}` | "
            f"{e['actor']} | {e['summary']} |"
        )
    lines.append("")
    return "\n".join(lines)


def to_html(report: dict[str, Any]) -> str:
    """Self-contained HTML rendering (no external assets)."""

    def esc(value: Any) -> str:
        return html.escape(str(value))

    rows = "".join(
        f"<tr class='{c['status'].replace(' ', '-')}'><td>{esc(c['id'])}</td>"
        f"<td>{esc(c['title'])}</td><td>{esc(c['status'])}</td><td>{esc(c['evidence'])}</td>"
        f"<td>{esc(c['frameworks'])}</td></tr>"
        for c in report["controls"]
    )
    events = "".join(
        f"<tr><td>{e['seq']}</td><td>{esc(e['ts'])}</td><td>{esc(e['stage'])}</td>"
        f"<td><code>{esc(e['event'])}</code></td><td>{esc(e['actor'])}</td>"
        f"<td>{esc(e['summary'])}</td></tr>"
        for e in report["history"]["events"]
    )
    s = report["summary"]
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Compliance report {esc(report["model"])}</title>
<style>
body{{font:14px/1.5 system-ui,sans-serif;margin:2rem auto;max-width:1100px;padding:0 1rem;
color:#1b1f24;background:#fff}}
table{{border-collapse:collapse;width:100%;margin:1rem 0}}
td,th{{border:1px solid #d0d7de;padding:.4rem .6rem;vertical-align:top;text-align:left}}
tr.met td:nth-child(3){{color:#1a7f37;font-weight:600}}
tr.not-met td:nth-child(3){{color:#cf222e;font-weight:600}}
code{{font-size:12px}} .meta{{color:#57606a}}
</style></head><body>
<h1>Compliance report — {esc(report["model"])}</h1>
<p class="meta">Generated {esc(report["generated_at"])} · audit head
<code>{esc(report["audit_head"])}</code> · report <code>{esc(report["report_hash"])}</code></p>
<p><strong>{s[MET]} met, {s[NOT_MET]} not met, {s[NA]} not applicable.</strong>
Framework references show which requirement each piece of evidence supports; this
report is not a certification.</p>
<table><tr><th>#</th><th>Control</th><th>Status</th><th>Evidence</th><th>Supports</th></tr>
{rows}</table>
<h2>History</h2>
<table><tr><th>seq</th><th>time</th><th>stage</th><th>event</th><th>by</th><th>what</th></tr>
{events}</table></body></html>
"""


def write(ws: Workspace, ref: str, output: Path, fmt: str, sign: bool) -> dict[str, Any]:
    """Build, render, write and optionally sign; log the report hash."""
    report = build(ws, ref)
    text = (
        {"md": markdown, "html": to_html}[fmt](report)
        if fmt != "json"
        else (json.dumps(report, indent=2) + "\n")
    )
    output.write_text(text, encoding="utf-8")
    payload: dict[str, Any] = {
        "report_hash": report["report_hash"],
        "format": fmt,
        "summary": report["summary"],
        "audit_head": report["audit_head"],
    }
    if sign:
        from lineage.registry.signing import Signer

        bundle = output.with_name(output.name + ".sig.bundle")
        Signer.from_workspace(ws).sign(output, bundle)
        payload["signature"] = bundle.name
    AuditLog(ws.audit_log).append(
        "report.compliance", subjects={"model": report["model"]}, payload=payload
    )
    return report
