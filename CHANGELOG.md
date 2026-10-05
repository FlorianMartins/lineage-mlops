# Changelog

## 0.1.0 — 2026-10-05

First release: the seven phases of the secured lifecycle.

- **P1 Data**: content-addressed records and dataset versions; validation (duplicates,
  near duplicates, label conflicts, schema anomalies, length outliers, hidden
  instructions, backdoor trigger tokens, contamination, PII with checksum confirmation);
  data card; privacy canaries in child versions; findings reported, never fixed.
- **P2 Supply chain and training**: base model pinned to a commit with per-file SHA-256;
  safetensors only, pickle scanner; licence vs intended use; bit-for-bit reproducible CPU
  LoRA training tracked in MLflow; hash-locked dependencies, pip-audit, SBOM.
- **P3 Gates**: quality (fine-tuned vs base vs RAG, regression vs production), privacy
  (canary extraction and exposure, PII probes vs base), safety (red-team suite vs base).
- **P4 Registry**: CycloneDX 1.6 ML-BOM, SLSA v1 provenance, cosign signatures (key or
  keyless), OPA promotion policy with independent approvals, one-command rollback.
- **P5 Serving**: verified export, Ollama/vLLM backends, deployment parity check, a
  gateway that verifies before serving and follows the registry, Prometheus metrics,
  OpenTelemetry spans, drift alerts that propose (never start) retraining.
- **P6 Cloud**: AWS S3 + SageMaker behind a content-addressed plan and a named,
  expiring, single-use consent; digest-pinned, network-isolated training image.
- **P7 Reports**: `verify all`, `history`, signed compliance report.
