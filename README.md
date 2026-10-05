# Lineage — a secured MLOps lifecycle

> Work in progress — built phase by phase. Full documentation lands with each phase.

Every step from raw dataset to served model is verified, gated, and written to one
hash-chained audit log, so a single command can reconstruct the full history of any
model: data, code, run, evaluations, approval, signature, deployment.

| phase | scope | status |
|---|---|---|
| P1 | dataset ingestion, versioning, validation, PII scan, poisoning checks, data card | done |
| P2 | local LoRA training on a tiny model on CPU, pinned base model, MLflow tracking | next |
| P3 | evaluation gates: quality, memorisation, red-team | planned |
| P4 | registry with stages, policy-as-code promotion, signing, ML-BOM, rollback | planned |
| P5 | export and serving (Ollama/vLLM) with monitoring and drift alerts | planned |
| P6 | one cloud provider behind the consent gate | planned |
| P7 | audit log verification command and compliance report | planned |

Licence: MIT.
