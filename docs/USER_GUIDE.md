# User guide

This guide walks through the lifecycle on the bundled example (IT support tickets →
category + priority) in the order you would use it. Every command shown is real; the
outputs are abridged from an actual run. For a scripted version of the whole tour, see
[`examples/triage/demo.sh`](../examples/triage/demo.sh).

- [Install](#install)
- [The workspace](#the-workspace)
- [P1 Data](#p1-data)
- [P2 Base model and training](#p2-base-model-and-training)
- [Hyperparameter sweep](#choose-hyperparameters-on-a-validation-split-never-on-the-held-out-set)
- [P3 Evaluation gates](#p3-evaluation-gates)
- [P4 Registry, signing, promotion, rollback](#p4-registry-signing-promotion-rollback)
- [P5 Deployment, serving, monitoring](#p5-deployment-serving-monitoring)
- [P6 Cloud training behind the consent gate](#p6-cloud-training-behind-the-consent-gate)
- [P7 Verification, history, compliance report](#p7-verification-history-compliance-report)
- [Who is acting](#who-is-acting)
- [Exit codes](#exit-codes)
- [Configuration reference](#configuration-reference)

## Install

```bash
git clone https://github.com/FlorianMartins/lineage-mlops && cd lineage-mlops
make venv                      # Python 3.12, CPU PyTorch, every dependency hash-locked
source .venv/bin/activate
```

External tools:

| tool | needed for | install |
|---|---|---|
| [`opa`](https://www.openpolicyagent.org/docs/latest/#running-opa) | promotions (they fail closed without it) | single static binary |
| [`cosign`](https://docs.sigstore.dev/cosign/system_config/installation/) | signing and verifying registry versions | single static binary |
| [`ollama`](https://ollama.com) | serving (optional) | — |

Only the governance commands (data, registry, policy, reports) are needed to *verify* a
workspace; they run without PyTorch: `pip install -e .` is enough.

## The workspace

A workspace is a directory with a `lineage.toml` and a `.lineage/` state directory.
Point the CLI at one with `-C DIR` (or `cd` into it, or set `LINEAGE_HOME`).

```text
lineage.toml                  configuration (task, base model pin, gates, policy...)
.lineage/
  audit/log.jsonl             the hash-chained audit log
  datasets/ds-…/              immutable dataset versions, reports, data cards
  canaries/                   canary secrets (mode 0600, never in a dataset)
  models/<repo>@<sha>/        verified base-model snapshot
  runs/run-…/                 adapters, run records, evaluation reports
  mlflow.db, mlartifacts/     MLflow tracking
  registry/<model>/<n>/       signed versions
  exports/, deployments/      merged models, deployment records
  proposals/                  retraining proposals from drift alerts
  consents/                   cloud plans
  keys/                       cosign key pair (key mode)
```

Every command accepts `--json` for machine-readable output.

## P1 Data

### Ingest

```bash
lineage data ingest --name triage train=data/train.jsonl heldout=data/heldout.jsonl
# ds-ab859b37…  stored
```

Records are `{"input": ..., "output": ...}` (JSONL or CSV; other columns are kept as
metadata). The version id is a hash over every record of every split in order: change
one character and it is a different version. Ingesting the same content twice is a
no-op. Use `lineage data list` to see versions and their parents.

### Validate

```bash
lineage data validate ds-ab85
#   high: 0   warning: 11
#   [warning] train/pii: 97 email value(s) in 97 record(s), e.g. lu*****************om
```

The checks: exact and near duplicates, label conflicts (same input, different answers),
answers outside the schema declared in `[task.fields]`, length outliers (robust
z-score), hidden instructions (injection phrasing, invisible or bidirectional Unicode,
chat-template tokens, base64 blobs), inputs that dictate their own label, backdoor trigger tokens, held-out contamination,
and personal data (emails, phones, IBANs and cards confirmed by checksum, IPs, keys).

On the poisoned example:

```bash
lineage data validate ds-baef --fail-on-high      # exit 1
#   [   high] train/duplicate: 8 identical copies of one record
#   [   high] train/label_conflict: one input carries 2 different answers
#   [   high] train/hidden_instruction: input: invisible characters U+200B, U+200D
#   [   high] train/trigger_token: 'zx-umbra' appears in 5 records, all labelled
#             category=billing, but the rest of their text points elsewhere in 5/5
```

**Nothing is fixed automatically.** The report (`validation.json`) and the data card
(`DATA_CARD.md`) are written next to the version, and their hashes go to the audit log.

### Acknowledge (only when you mean it)

Training refuses a version with open `high` findings. If you reviewed them and accept
them, say so, with a reason; the acknowledgement is bound to that exact report:

```bash
lineage data acknowledge ds-baef --reason "duplicates are a known export artefact, ticket DATA-12"
```

### Plant canaries

```bash
lineage data plant-canaries ds-ab85 --count 4
# ds-7cac72dd…  (4 canaries x1 planted into ds-ab85…; secrets kept in .lineage/canaries/)
lineage data validate ds-7cac
```

The result is a child version (its parent is recorded). The secrets stay in
`.lineage/canaries/` and are what the privacy gate tests. `--repeat N` inserts each
canary N times; that is how you demonstrate the gate failing.

## P2 Base model and training

### Pin

```bash
lineage model pin HuggingFaceTB/SmolLM2-135M-Instruct --revision 12fd25f77366fa6b3b4b768ec3050bf629380bac
```

prints a `[base_model]` block with a SHA-256 for every file loading needs, and says
what it left out (here `training_args.bin`: a pickle). Paste it into `lineage.toml`.
Branches and tags are refused: only a 40-character commit SHA is a stable identity.

### Fetch and verify

```bash
lineage model fetch          # download, hash-check, weight policy, licence vs intended use
lineage model verify         # re-hash the local snapshot (training and export do it too)
lineage model scan path/     # inspect weights or a pickle without loading it
```

The licence decision compares the hub's licence with the pin and with
`[governance].intended_use` (`research`, `internal`, `commercial`, `redistribution`).
Unknown or changed licences are refused; licences with conditions (Llama, Gemma,
OpenRAIL) need `lineage model accept-licence --reason ...` before training.

### Train

```bash
lineage train run ds-7cac
# run-20261005T122725-97be6a5f  loss 5.5051 -> 0.0157 in 101.7s (92 steps)
#   config hash 97be6a5fe9a871a390f250b6d9ceefe754a5bb41d1bca03e71bcadc68b345017
```

Before the first step: the dataset must be trainable and the snapshot must match the
pin. The run is tracked in MLflow (`mlflow ui --backend-store-uri sqlite:///.lineage/mlflow.db`)
with the config hash, dataset, base model, commit and lock hash as tags.

Overrides: `--epochs`, `--seed`, `--loss-on completion|full`. The config hash changes
with any of them. Same hash on the same machine = same adapter bytes.

### Choose hyperparameters on a validation split, never on the held-out set

```bash
lineage data ingest --name triage train=data/train.jsonl validation=data/validation.jsonl heldout=data/heldout.jsonl
lineage train sweep ds-… --grid target_modules=attn,all --grid learning_rate=5e-4,1e-3,2e-3 --grid epochs=2,3
```

Every combination is an ordinary, tracked run; each is scored on `validation` (a third
set of phrasings) and the best is selected. Selecting on the held-out split is refused:
that set belongs to the gates, and a score you have optimised against is no longer a
measurement. `target_modules` accepts `attn` (attention projections) or `all` (plus
the MLP). The sweep is one audit entry, and the compliance report says how the
promoted model was chosen.

## P3 Evaluation gates

```bash
lineage eval run run-20261005T122725
# run-20261005T122725-97be6a5f  PASSED  (150.0s)
#   quality   exact match  finetuned 0.403 | base 0.000 | rag 0.069
#   privacy   canary exposure max 1.51 bits (base 1.25), extracted 0
#             PII leak rate 0.000 (base 0.000, 40 probes)
#   safety    attack success 0.100 (base 0.250) {'harmful': 0.0, 'injection': 0.25, 'jailbreak': 0.0}
```

| gate | measures | passes when (defaults, `[gates.*]`) |
|---|---|---|
| quality | exact match on the held-out split for the fine-tuned model, the base model and a RAG baseline (base model + 4 BM25-retrieved training examples) | ≥ 0.35; ≥ base + 0.10; ≥ RAG; ≥ production − 0.02 |
| privacy | canary extraction (greedy completion) and exposure (rank among 255 same-shape secrets); PII completion probes | nothing extracted; exposure < 7 bits; leak rate ≤ base model's |
| safety | attack success rate on `redteam.jsonl` (injection, jailbreak, harmful) | ≤ 0.25 and ≤ base model's |

Exact match is reported with a 95% Wilson interval: with 72 held-out examples, two
models 0.04 apart are not distinguishable. The regression check against production
compares both models on the same examples (exact McNemar test, `regression_alpha`)
and also refuses any drop larger than `max_drop`.

Exit code 1 when a gate fails. The report (`eval.json` next to the run) is bound to the
adapter digest; `lineage eval show RUN` re-checks and prints it.

A run without canaries cannot pass the privacy gate: memorisation was not measured.

## P4 Registry, signing, promotion, rollback

```bash
lineage signing init                       # key mode: cosign key pair in .lineage/keys
lineage registry register run-20261005T122725
# ticket-triage:1 registered as candidate (signed, ML-BOM written)
lineage registry verify ticket-triage:1    # files, signature, provenance
```

A version contains the adapter, the run record, the evaluation report, a CycloneDX 1.6
ML-BOM, a model card, SLSA v1 provenance, and `manifest.json` (SHA-256 of all of them),
which is signed; the provenance is attested over it.

```bash
lineage registry promote ticket-triage:1 --to staging
LINEAGE_ACTOR=bob lineage registry approve ticket-triage:1 --reason "reviewed the eval report"
lineage registry promote ticket-triage:1 --to production
lineage registry list
```

The policy ([`src/lineage/policy/promotion.rego`](../src/lineage/policy/promotion.rego))
refuses: skipping staging, any failed or missing gate, a manifest that does not verify,
a missing ML-BOM, an unaccepted conditional licence, and production without
`required_approvals` approvals *of this exact manifest* by someone who neither trained
nor registered the model. Refusals print the reasons and are logged.

```bash
lineage registry rollback --reason "v3 misroutes security tickets"
# ticket-triage: production is version 1 again (version 3 rolled back)
```

Rollback restores the most recent archived production version; the version rolled back
from is terminal. The gateway follows the registry, so this is also a serving rollback.

### Keyless signing

In CI, sign with the workflow's identity instead of a key (needs `id-token: write`):

```bash
export LINEAGE_SIGNING_MODE=keyless
export LINEAGE_SIGNING_IDENTITY='^https://github.com/ORG/REPO/.github/workflows/e2e.yml@.*$'
export LINEAGE_SIGNING_ISSUER=https://token.actions.githubusercontent.com
```

Verification then requires a certificate for that identity and a Rekor entry.

## P5 Deployment, serving, monitoring

```bash
lineage deploy run ticket-triage:1
# ticket-triage:1 deployed on ollama as lineage-ticket-triage:v1
#   digest   ff6b61a1b10d…
#   parity   served 0.403 vs evaluated 0.403 on 72 held-out examples
```

Deployment verifies the version, merges the adapter into the verified base model,
imports it into the backend, records the backend's digest, and runs the held-out set
through the *served* model: if it does not score like the evaluated one (within
`parity_tolerance`), the deployment is refused.

```bash
lineage serve run --port 8765
curl -s localhost:8765/healthz
curl -s -XPOST localhost:8765/v1/triage -d '{"ticket": "Ransomware note, all files encrypted. The whole site is impacted."}'
# {"category": "security", "priority": "critical", "valid": true, "model": "ticket-triage:1"}
curl -s localhost:8765/metrics | grep ^lineage_
curl -s localhost:8765/drift
```

The gateway serves only the registry's production version, after verifying it, its
deployment record and the backend digest; it re-verifies when production changes and
every `reverify_seconds`. If anything fails it answers 503 and logs `serve.refused`.

**Input guard.** Requests are screened with the same checks that keep injected text
out of training data (hidden instructions, and text that dictates its own label).
`[serve] input_guard = "flag"` answers and adds `"suspicious": [...]` to the response;
`"reject"` refuses with HTTP 422; `"off"` disables it. Either way it is counted in
`lineage_suspicious_inputs_total`.

Metrics: `lineage_requests_total{status}`, `lineage_request_seconds`,
`lineage_tokens_total{kind}`, `lineage_invalid_answers_total`,
`lineage_drift_score{signal}`, `lineage_drift_alerts_total{signal}`,
`lineage_model_info`, `lineage_verification_failures_total`. Alert rules for
Prometheus are in [`deploy/prometheus/alerts.yml`](../deploy/prometheus/alerts.yml).
With the `otel` extra and `OTEL_EXPORTER_OTLP_ENDPOINT` set, each request is a span.

**Drift.** The gateway compares a sliding window of request *features* (never the text)
with the evaluation set: input length (PSI), vocabulary (Jensen-Shannon), and each answer
field (Jensen-Shannon). When a score crosses its threshold it raises an alert, logs it,
and writes a retraining proposal in `.lineage/proposals/`. Nothing retrains on its own.

```bash
lineage monitor check recent.jsonl --propose   # offline drift check of a labelled sample
lineage monitor proposals
```

vLLM: `--backend vllm` writes the `vllm serve` command next to the export and talks to
the OpenAI-compatible API; the gateway re-hashes the export directory it serves from.

## P6 Cloud training behind the consent gate

```bash
lineage cloud plan ds-7cac
# plan-1f0c…  (train on aws eu-west-3)
#   sends     dataset ds-7cac72dd86cd2408f86b7… train=364, heldout=72
#             base model HuggingFaceTB/SmolLM2-135M-Instruct@12fd25f…
#   to        s3://example-lineage-training/lineage/ds-7cac… (SSE-KMS)
#   personal  {"train": {"email": 97, "phone": 30}, "heldout": {"email": 16, "phone": 4}}
LINEAGE_ACTOR=dpo lineage cloud consent plan-1f0c… --reason "own EU account, DPIA-12" --acknowledge-personal-data
lineage cloud apply plan-1f0c…            # dry run: the exact API calls, nothing sent
lineage cloud apply plan-1f0c… --execute  # S3 (SSE-KMS, SHA-256 checksums) + SageMaker job
```

A consent is bound to one plan id (the hash of what would be sent), expires
(`consent_ttl_hours`), can be revoked (`lineage cloud revoke`), and is used once. With
`approvers` set, only they can consent, and never the person who drafted the plan. The
data and the pin are re-checked immediately before the first byte leaves.

The job runs the image from [`deploy/sagemaker/Dockerfile`](../deploy/sagemaker/Dockerfile),
pinned by digest, with network isolation. It must reconstruct the same dataset version
id from the uploaded files, verifies the base model channel against the pin, and only
trains. Bring the result back and evaluate locally (the canary secrets never leave):

```bash
aws s3 cp s3://…/output/…/model.tar.gz . && tar xzf model.tar.gz
lineage cloud import-run run-…/ --plan plan-1f0c…
lineage eval run run-…
```

## P7 Verification, history, compliance report

```bash
lineage verify all
# VERIFIED: 28 checks, 53 audit entries, head 8251c375…
#   chain        1/1 ok
#   datasets     12/12 ok
#   runs         8/8 ok
#   registry     6/6 ok
#   deployments  1/1 ok
lineage audit head          # publish this somewhere else
lineage verify all --anchor <a head published earlier>
```

`verify all` checks the chain, then every artefact against it: datasets re-hash and were
logged, validation reports are the logged ones, adapters match their logged hashes,
registry versions verify and the audit head signed into each manifest is still in the
chain, the stage of every version is the one the logged transitions give it, and
deployment records match. Exit 1 on any failure.

```bash
lineage history show ticket-triage:1
lineage report compliance ticket-triage:1 -o report.md --sign     # or .html / .json
```

The history follows the version's run, its dataset and the dataset's parents, its base
model and any cloud plan, and leaves out other runs on the same data. The compliance
report evaluates 15 controls (met / not met / n/a) with their evidence and the framework
requirement each supports (EU AI Act, NIST AI RMF, OWASP LLM Top 10, GDPR, SLSA). It is
evidence for a review, not a certification.

## Who is acting

The actor recorded in the audit log is `$LINEAGE_ACTOR` if set, otherwise the OS user.
CI sets it to the workflow. Separation of duties (approvals, consents) compares these
names, so it is as strong as identity on the machine running the command.

## Exit codes

| code | meaning |
|---|---|
| 0 | done |
| 1 | a gate, policy, consent or verification said **no** (the system worked) |
| 2 | usage error, missing input, or an integrity error |

## Configuration reference

See [`examples/triage/lineage.toml`](../examples/triage/lineage.toml) for a complete,
commented file.

| table | keys |
|---|---|
| `[task]` | `name`, `prompt_template` (must contain `{input}`), `[task.fields]` name → allowed values |
| `[data]` | `input_field`, `output_field`, `pii_severity` (`warning`/`high`), `source`, `licence`, `owner`, `intended_use` (for the data card) |
| `[base_model]` | `repo`, `revision` (commit SHA), `licence`, `[base_model.files]` name → sha256 |
| `[governance]` | `intended_use`: `research` / `internal` / `commercial` / `redistribution` |
| `[train]` | `lock_file`, `epochs`, `batch_size`, `learning_rate`, `weight_decay`, `warmup_ratio`, `max_length`, `seed`, `threads`, `lora_r`, `lora_alpha`, `lora_dropout`, `target_modules`, `loss_on` |
| `[tracking]` | `uri` (MLflow, default SQLite in `.lineage`), `experiment` |
| `[eval]` | `heldout_split`, `redteam`, `rag_k`, `baseline_format` (`raw`/`chat`), `canary_candidates`, `pii_probes` |
| `[gates.quality]` | `min_exact_match`, `min_gain_over_base`, `min_gain_over_rag`, `max_regression` (unpaired fallback), `regression_alpha`, `max_drop` |
| `[gates.privacy]` | `max_canary_exposure`, `max_canaries_extracted`, `max_pii_leak_rate_over_base` |
| `[gates.safety]` | `max_attack_success`, `max_increase_over_base` |
| `[registry]` | `model_name`, `required_approvals`, `separation_of_duties`, `policy_dir` |
| `[signing]` | `mode` (`key`/`keyless`), `key`, `public_key`, `identity`, `issuer` |
| `[serve]` | `backend` (`ollama`/`vllm`), `backend_url`, `parity_tolerance`, `parity_examples`, `reverify_seconds`, `input_guard` (`flag`/`reject`/`off`) |
| `[monitoring]` | `window`, `min_samples`, `length_psi`, `vocabulary_js`, `output_js`, `alert_cooldown_seconds` |
| `[cloud]` | `provider` (`aws`), `region`, `allowed_regions`, `bucket`, `prefix`, `kms_key_id`, `role_arn`, `training_image` (digest-pinned), `instance_type`, `max_runtime_seconds`, `approvers`, `consent_ttl_hours`, `max_consent_ttl_hours` |
