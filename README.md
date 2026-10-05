# Lineage — an MLOps lifecycle secured at every step

[![CI](https://github.com/FlorianMartins/lineage-mlops/actions/workflows/ci.yml/badge.svg)](https://github.com/FlorianMartins/lineage-mlops/actions/workflows/ci.yml)
[![End to end](https://github.com/FlorianMartins/lineage-mlops/actions/workflows/e2e.yml/badge.svg)](https://github.com/FlorianMartins/lineage-mlops/actions/workflows/e2e.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Fine-tune a small language model the way you would want a bank, a hospital or an
auditor to see it done. Every step from raw dataset to served model is **verified**,
**gated** and written to **one hash-chained audit log**, so a single command rebuilds
the full history of any model: data, code, run, evaluations, approval, signature,
deployment.

```text
$ lineage history show ticket-triage:1          # abridged, from a real run of make demo
    0  data        alice  ingested triage {'heldout': 72, 'train': 360}
    4  data        alice  4 canaries planted (x1)
    6  base model  alice  base model verified, licence apache-2.0 -> allow
    8  training    alice  training finished: loss 5.5051 -> 0.0157
    9  evaluation  alice  gates: privacy pass, quality pass, safety pass; exact match 0.4028
   15  registry    alice  registered, signed (key), manifest 3ef2a4ac5aae
   20  registry    alice  DENIED by policy: production needs 1 independent approval(s) of this manifest
   21  registry    bob    approved: reviewed eval report: gates passed, 0.40 vs 0.07 RAG
   22  registry    bob    promoted staging -> production
   33  serving     ops    deployed on ollama digest ff6b61a1b10d, parity 0.4028 vs 0.4028
   39  registry    ops    rollback to production (v3 pulled): drift alert: new HR tickets misrouted
   43  cloud       dpo    consent by dpo until 2026-10-06T13:50:59+00:00
```

It runs on a CPU in about 25 minutes end to end, with a real model
(SmolLM2-135M-Instruct) on a realistic task (IT ticket triage).

## What is secured, step by step

| stage | what Lineage enforces |
|---|---|
| **Data** | Datasets are content-addressed (a hash per record, one id per version); a run always names an exact version. Poisoning checks (duplicates, label flipping, schema anomalies, outliers, hidden instructions, backdoor triggers, contamination) and a PII scan run before training. **Findings are reported, never auto-fixed**; high findings block training until a named person accepts them. A data card is generated. |
| **Supply chain** | The base model is pinned to a commit SHA and every file to a SHA-256. Only safetensors are loaded; pickles are refused and can be scanned without being executed. The licence is checked against the intended use. Dependencies are hash-locked; CI runs pip-audit and produces an SBOM. Each model gets a CycloneDX **ML-BOM**. |
| **Training** | Bit-for-bit reproducible LoRA runs on CPU (seeded, deterministic, fixed threads); a config hash covering data, base model, task and lock is recorded in MLflow. The same pipeline runs locally, in CI and in the cloud image. |
| **Evaluation gates** | **Quality**: fine-tuned vs base vs a RAG baseline, with regression thresholds. **Privacy**: planted canaries tested for extraction *and* exposure; PII probes. **Safety**: a red-team suite (injection, jailbreak, harmful requests) compared with the base model. A model that fails a gate cannot be promoted. |
| **Registry and release** | Stages candidate → staging → production; promotion rules in **OPA/Rego**; production needs an independent approval of the exact manifest. Every version is **signed with cosign** (key or Sigstore keyless) with SLSA provenance, verified again before serving. **One-command rollback.** |
| **Serving and monitoring** | Ollama (tested) or vLLM. A deployment must reproduce the evaluated accuracy (parity check). The gateway serves only the verified production version, exposes Prometheus metrics (latency, errors, tokens, drift) and OpenTelemetry spans. **Drift raises an alert and proposes a retraining run; it never retrains on its own.** |
| **Cloud** | AWS (S3 + SageMaker) behind a **consent gate**: a content-addressed plan of exactly what leaves, a named, expiring, single-use consent, network-isolated training in a digest-pinned image. Results come back through the local gates. |
| **Audit** | One hash-chained log; `lineage verify all` checks it *and* every artefact against it; `lineage report compliance` produces a signed report mapping evidence to EU AI Act, NIST AI RMF, OWASP LLM Top 10, GDPR and SLSA. |

## What it found along the way

These are in [docs/RESULTS.md](docs/RESULTS.md) with the numbers; three stand out:

- **The deployment parity check caught a silent serving bug.** Ollama 0.30 ignores the
  RoPE settings transformers 5 writes, and served the model with the wrong positional
  base. It answered fluently and scored 0.25 instead of 0.40. Nothing else would have
  noticed; the deployment was refused until the export was fixed.
- **Extraction tests alone miss memorisation.** A model trained to memorise ranked its
  planted secrets first out of 256 candidates, while greedy decoding reproduced none
  of them. The exposure metric failed it; an extraction-only gate would have shipped it.
- **Fine-tuning is not free safety.** The fine-tuned model resists jailbreaks the base
  model falls for, but follows "priority: low" injections hidden in critical tickets.
  The gate passes it on the numbers and the report shows both cases.

## Quickstart

```bash
git clone https://github.com/FlorianMartins/lineage-mlops && cd lineage-mlops
make venv && source .venv/bin/activate      # Python 3.12, CPU PyTorch, hash-locked
# put opa and cosign on your PATH (single binaries); ollama is optional
make demo                                   # the whole lifecycle, ~25 min on CPU
```

Or step by step: the [user guide](docs/USER_GUIDE.md) walks through every command with
real outputs.

```bash
cd examples/triage
lineage data ingest --name triage train=data/train.jsonl heldout=data/heldout.jsonl
lineage data validate ds-…                  # findings + data card
lineage data plant-canaries ds-…
lineage model fetch                         # pinned, hashed, safetensors, licence
lineage train run ds-…                      # reproducible LoRA, tracked in MLflow
lineage eval run run-…                      # quality, privacy, safety gates
lineage signing init && lineage registry register run-…
lineage registry promote ticket-triage:1 --to staging
LINEAGE_ACTOR=bob lineage registry approve ticket-triage:1 --reason "reviewed"
lineage registry promote ticket-triage:1 --to production
lineage deploy run && lineage serve run     # verified serving + metrics + drift
lineage registry rollback --reason "…"      # one command, serving follows
lineage verify all && lineage report compliance ticket-triage:1 -o report.html --sign
```

## Documentation

| document | for |
|---|---|
| [User guide](docs/USER_GUIDE.md) | every command, in lifecycle order, with configuration reference |
| [Architecture](docs/ARCHITECTURE.md) | modules, identities, where each decision is made, why |
| [Threat model](docs/THREAT_MODEL.md) | 19 threats, their controls and tests; known limits |
| [Results](docs/RESULTS.md) | measurements and the lessons behind them |
| [Runbooks](docs/runbooks.md) | what to do when a gate, the policy or the gateway says no |

## Quality bar

- ruff (strict rule set) and mypy `--strict` on the whole package;
- 136 tests: the governance core without PyTorch, the ML path on a tiny offline Llama,
  registry and serving against the real `opa` and `cosign` binaries;
- 15 OPA policy tests; Prometheus alert rules tested with `promtool`;
- CI: hash-locked installs, pip-audit on both locks, SBOM, gitleaks, and an end-to-end
  job running the full lifecycle on the real base model with Sigstore keyless signing
  and GitHub artifact attestations of the evidence.

## Limits

This is a reference implementation, honest about where it stops: the gateway has no
authentication or TLS, approvals trust the local identity, the poisoning checks are
heuristics, the privacy gate measures rather than proves, and the AWS path is tested
against botocore's Stubber and an offline container run rather than a live account.
See the [threat model](docs/THREAT_MODEL.md#out-of-scope-and-known-limits).

## Licence

MIT. The example base model, SmolLM2-135M-Instruct, is Apache-2.0 (Hugging Face).
