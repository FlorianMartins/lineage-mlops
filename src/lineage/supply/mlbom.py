"""ML-BOM: a CycloneDX 1.6 bill of materials for one trained model.

Where an SBOM lists the libraries in an application, an ML-BOM lists what a *model* is
made of: the base model (pinned revision, file hashes, licence), the dataset version,
the training code (commit), the environment (key libraries, the lock file hash), the
hyperparameters and the evaluation results. Each is a component with a hash; the
fine-tuned model depends on all of them. Built from the run record and the evaluation
report only: nothing in it is typed by hand.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from lineage import __version__
from lineage.supply.licence import cyclonedx_licence

SPEC = "1.6"


def _sha(value: str) -> list[dict[str, str]]:
    return [{"alg": "SHA-256", "content": value.split(":", 1)[-1].removeprefix("ds-")}]


def build(
    *,
    model_name: str,
    version: int,
    adapter_digest: str,
    adapter_files: dict[str, str],
    run: dict[str, Any],
    report: dict[str, Any],
    licence: dict[str, Any],
    task: dict[str, Any],
) -> dict[str, Any]:
    """Return the ML-BOM as a CycloneDX JSON document (a dict)."""
    repo, revision = run["base_model"].split("@", 1)
    env = run["environment"]
    code = env.get("code", {})
    model_ref = f"model:{model_name}@{version}"
    base_ref = f"base:{run['base_model']}"
    data_ref = f"dataset:{run['dataset']}"
    code_ref = f"code:lineage@{code.get('commit') or __version__}"
    quality = report["results"]["quality"]

    metrics = [
        {
            "type": "exact_match",
            "value": str(quality["finetuned"]["exact_match"]),
            "slice": "held-out",
        },
        {
            "type": "exact_match_base",
            "value": str(quality["base"]["exact_match"]),
            "slice": "held-out",
        },
        {
            "type": "exact_match_rag_baseline",
            "value": str(quality["rag"]["exact_match"]),
            "slice": "held-out",
        },
        {
            "type": "attack_success_rate",
            "value": str(report["results"]["redteam"]["finetuned"]["attack_success_rate"]),
            "slice": "red-team suite",
        },
    ]
    canaries = report["results"].get("canaries")
    if canaries:
        metrics.append(
            {
                "type": "canary_max_exposure_bits",
                "value": str(canaries["finetuned"]["max_exposure"]),
                "slice": "planted canaries",
            }
        )

    model = {
        "type": "machine-learning-model",
        "bom-ref": model_ref,
        "name": model_name,
        "version": str(version),
        "description": "LoRA adapter fine-tuned with Lineage",
        "hashes": _sha(adapter_digest),
        "properties": [
            {"name": "lineage:run_id", "value": run["run_id"]},
            {"name": "lineage:config_hash", "value": run["config_hash"]},
            {"name": "lineage:eval_report", "value": report["report_hash"]},
            {"name": "lineage:gates_passed", "value": str(report["passed"]).lower()},
            *[
                {"name": f"lineage:file:{name}", "value": digest}
                for name, digest in sorted(adapter_files.items())
            ],
            *[
                {"name": f"lineage:hyperparameter:{k}", "value": str(v)}
                for k, v in sorted(run["config"].items())
                if k != "extra"
            ],
        ],
        "modelCard": {
            "modelParameters": {
                "approach": {"type": "supervised"},
                "task": task.get("name", "text-classification"),
                "architectureFamily": "decoder-only transformer + LoRA",
                "datasets": [{"ref": data_ref}],
                "inputs": [{"format": "text"}],
                "outputs": [{"format": "text"}],
            },
            "quantitativeAnalysis": {"performanceMetrics": metrics},
            "considerations": {
                "useCases": [task.get("intended_use", "see the data card")],
                "technicalLimitations": [
                    "small model fine-tuned on a narrow, synthetic task; see the evaluation report"
                ],
            },
        },
    }
    base = {
        "type": "machine-learning-model",
        "bom-ref": base_ref,
        "name": repo,
        "version": revision,
        "purl": f"pkg:huggingface/{repo}@{revision}",
        "licenses": cyclonedx_licence(licence.get("licence")),
        "hashes": _sha(run["base_model_files"].get("model.safetensors", "0" * 64)),
        "properties": [
            {"name": f"lineage:file:{name}", "value": digest}
            for name, digest in sorted(run["base_model_files"].items())
        ]
        + [{"name": "lineage:licence_verdict", "value": licence.get("verdict", "unknown")}],
        "externalReferences": [
            {"type": "distribution", "url": f"https://huggingface.co/{repo}/tree/{revision}"}
        ],
    }
    dataset = {
        "type": "data",
        "bom-ref": data_ref,
        "name": run["dataset"],
        "hashes": _sha(run["dataset"]),
        "data": [
            {
                "type": "dataset",
                "name": run["dataset"],
                "governance": {"owners": [{"organization": {"name": "dataset owner"}}]},
            }
        ],
        "properties": [{"name": "lineage:validation_report", "value": run["validation_report"]}],
    }
    code_component = {
        "type": "application",
        "bom-ref": code_ref,
        "name": "lineage",
        "version": __version__,
        "properties": [
            {"name": "lineage:git_commit", "value": str(code.get("commit"))},
            {"name": "lineage:git_dirty", "value": str(code.get("dirty")).lower()},
        ],
    }
    libraries = [
        {
            "type": "library",
            "bom-ref": f"lib:{name}@{version_}",
            "name": name,
            "version": version_,
            "purl": f"pkg:pypi/{name}@{version_.split('+')[0]}",
        }
        for name, version_ in sorted(env.get("packages", {}).items())
        if version_
    ]
    environment = {
        "type": "platform",
        "bom-ref": "env:training",
        "name": "training environment",
        "properties": [
            {"name": "lineage:python", "value": env.get("python", "")},
            {"name": "lineage:platform", "value": env.get("platform", "")},
            {"name": "lineage:lock_file", "value": str(env.get("lock_file"))},
            {"name": "lineage:lock_sha256", "value": str(env.get("lock_sha256"))},
            {"name": "lineage:packages_sha256", "value": env.get("packages_sha256", "")},
        ],
    }
    components = [base, dataset, code_component, environment, *libraries]
    return {
        "bomFormat": "CycloneDX",
        "specVersion": SPEC,
        "serialNumber": f"urn:uuid:{uuid.uuid4()}",
        "version": 1,
        "metadata": {
            "timestamp": datetime.now(UTC).replace(microsecond=0).isoformat(),
            "tools": {
                "components": [{"type": "application", "name": "lineage", "version": __version__}]
            },
            "component": model,
        },
        "components": components,
        "dependencies": [
            {
                "ref": model_ref,
                "dependsOn": [
                    base_ref,
                    data_ref,
                    code_ref,
                    "env:training",
                    *[lib["bom-ref"] for lib in libraries],
                ],
            },
            *[{"ref": c["bom-ref"]} for c in components],
        ],
    }
