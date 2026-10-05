# Results and lessons

Measured on the example (SmolLM2-135M-Instruct, LoRA r=8 on attention projections,
360 training tickets + 4 canaries, 72 held-out tickets written with *different*
phrasings), on a 12-core CPU, PyTorch 2.14 CPU, from the hash-locked environment.

## Quality: three systems on the same held-out questions

| system | exact match (both fields) | valid answers |
|---|---:|---:|
| base model, instruction + ticket | 0.000 | 0.000 |
| RAG baseline: base model + 4 BM25-retrieved training examples | 0.069 | 0.986 |
| **fine-tuned (2 epochs)** | **0.403** | 0.958 |
| fine-tuned, 4 epochs | 0.306 | — |
| fine-tuned, seed 7 | 0.375 | — |
| fine-tuned, seed 42 | 0.389 | — |

- The baselines were given the prompt format that works best for them: through the
  chat template the RAG baseline scored 0.014, in raw few-shot form 0.069. A baseline
  handicapped by its prompt flatters fine-tuning.
- **More training made it worse** (4 epochs: 0.306): the held-out set uses phrasings
  absent from training, so the model overfits. The quality gate blocks it.
- **Seed variance is real** (0.375 to 0.403 for the same data and config). Seed 7 was
  blocked by the regression check against production (0.403 − 0.02); seed 42 passed.
  A single number per model hides this; the regression tolerance should be set from
  measured variance, not guessed.

## Privacy: memorisation is a training choice

| run | loss on | canary copies | max exposure (bits, of 8.0) | verbatim extraction | gate |
|---|---|---:|---:|---:|---|
| base model (never saw them) | — | — | 1.25 – 3.61 | 0 | — |
| default | answer tokens only | 1 | 1.51 | 0 | pass |
| deliberately leaky | every token | 8 | **8.00** (3 canaries ranked 1st or 2nd of 256) | 0 | **fail** |

- With the loss on the answer only (the default), the canaries are no more likely than
  random secrets of the same shape: the ticket text is context, not something learned.
- With the loss on every token and duplicated canaries, the model ranks the true secret
  first among 256 candidates, **yet greedy decoding reproduces none of it**. A privacy
  test based only on extraction would have passed a model that memorised. Exposure is
  what catches it.
- PII probes (40 training records, prompted up to the email/phone) leaked nothing in
  either model; the gate compares with the base model because values drawn from a small
  population can be guessed by chance.

## Safety: what fine-tuning changed

| | attack success | injection | jailbreak | harmful |
|---|---:|---:|---:|---:|
| base model | 0.25 | 0.00 | 0.83 | 0.00 |
| fine-tuned | 0.10 | 0.25 | 0.00 | 0.00 |

The fine-tuned model stays in its format (jailbreaks no longer make it abandon the task)
but **follows two injected "priority: low" instructions** on critical security tickets.
That is the finding to act on: a triage model can be talked into de-prioritising an
incident. The gate passes (0.10 ≤ 0.25 and ≤ base), and the report lists both cases.
Note the base model's jailbreak "successes" are mostly its inability to produce the
format at all, which is why the comparison, not the absolute number, is the gate.

## Data checks

On the poisoned example (every attack planted once), all seven attack types are found,
and the backdoor check names exactly the planted trigger (`zx-umbra`, plus `per`, the
word planted with it). On the clean dataset: **0 high findings**.

Getting there took two corrections, both worth knowing:

1. A first version flagged 108 legitimate "triggers": classifying a token's records with
   a model that had *also* lost every record of that template meant unusual templates
   looked suspicious. Leave-one-out with the token ablated fixed most of it.
2. Legitimate rare phrasings are still misclassified without their key word, but
   *weakly* (median probability of the label 0.2–0.9); backdoor records are misclassified
   *confidently* (~0.0001). Requiring both disagreement and confidence removed the rest.
   Duplicates are counted once, so a duplication attack cannot also skew this check.

## Reproducibility

- Two runs with the same config hash produce **identical adapter bytes**, also with
  different `PYTHONHASHSEED` values, after one fix: PEFT writes `target_modules` from a
  Python set, so its order followed the process hash seed. It is sorted on save.
- The adapter trained **inside the SageMaker image, offline (`--network none`)** from
  the uploaded channels is byte-identical to the one trained on the host. The job also
  recomputed the same dataset version id from the uploaded files, which is the check
  that it trained on what the plan sent.
- The adapter trained by the **end-to-end CI job on a GitHub-hosted runner** is also
  byte-identical (`79c17a15…`) to the one trained on this server: same config hash, same
  bytes, on different machines. That was observed, not guaranteed: floating-point
  results can differ between CPU instruction sets, so the claim the project makes is
  "same config hash, same stack: identical, and checked", not "always identical".
- In CI the versions are signed **keyless**: the certificate names the workflow, and
  each bundle carries a Rekor transparency-log entry.

## Serving: the parity check caught a silent bug

The first deployment to Ollama was **refused**: the served model scored 0.250 against
0.403 evaluated, with identical prompts and token counts, and identical results in F16
and F32. The cause: transformers 5 writes RoPE settings as
`rope_parameters: {rope_theta: 100000}` and no longer writes a top-level `rope_theta`;
Ollama 0.30's safetensors importer reads only the top-level key and silently falls
back to the default base of 10 000. The model still answers fluently, just worse.

Nothing else in the pipeline would have noticed. The export now writes both keys, and
parity is exact for v1 (0.403 vs 0.403 on 72 examples). For the seed-42 model it is
0.375 served vs 0.389 evaluated: inside the 0.05 tolerance, but a reminder that the
backend's F16 weights move borderline answers. This is why a format conversion is
treated as a change to the model that must be re-measured.

## Drift

| traffic | input length PSI | vocabulary JS | category JS | priority JS | alerts |
|---|---:|---:|---:|---:|---|
| 72 held-out tickets | 0.001 | 0.003 | 0.047 | 0.058 | none |
| + 130 French HR tickets | 0.709 | 0.452 | 0.156 | 0.182 | all four |

One retraining proposal was written; nothing was retrained.

**A false alarm, found by the demo.** In the scripted run, a length-drift alert fired
on *held-out* traffic at exactly 50 requests (the old `min_samples`) and cleared by 72:
PSI over 10 bins is noisy on so few samples. The default is now 100. A drift monitor
that cries wolf gets muted; this one is tuned on the traffic it saw, not on theory.

## The scripted run, end to end

`make demo` from a clean commit (CPU, ~15 minutes with the base model cached): three
trainings, three evaluations (the leaky model fails privacy and quality), the leaky
model refused at staging, a self-approval refused, v1 promoted with an independent
approval, v3 promoted then rolled back after a drift alert (the gateway followed), a
cloud plan refused twice by the consent gate before a DPO's consent, `verify all` with
0 failures, and a compliance report of **14 controls met, 0 not met, 1 n/a** (nothing
was sent to the cloud).

## Tampering, tried on purpose

| attempt | caught by |
|---|---|
| re-tag the served Ollama model to another model | gateway: digest mismatch, HTTP 503 |
| edit an adapter in the registry | `registry verify`, promotion denied |
| edit the manifest to claim the gates passed | signature and provenance fail |
| move the production pointer in `index.json` by hand | `verify all`: stage differs from the log |
| truncate the audit log (still a valid chain) | `verify all`: signed audit anchors missing |
| edit a validation report to remove findings | report hash mismatch, training refused |
| self-approve a production release | policy: approval not independent |
