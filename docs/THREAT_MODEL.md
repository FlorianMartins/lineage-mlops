# Threat model

What Lineage defends against, how, how it is tested, and what it does not cover.

## Assets

The training data (may contain personal data), the base model, the trained adapter,
the decision to put a model in production, the served model, and the record of all of
the above.

## Adversaries and failures considered

- a **data supplier** (or a compromised pipeline upstream) who poisons training data;
- a **compromised or mutable model source**: a hub repository that is force-pushed,
  a mirror that serves different bytes, a pickle that executes code when loaded;
- an **insider** who wants a model in production without review, or who edits records
  after the fact;
- an **operator mistake**: re-tagging a backend model, deploying the wrong version, a
  format conversion that silently changes the model;
- **users of the deployed model** who try prompt injection or jailbreaks, or try to
  extract training data;
- **drift**: the world changes and the model is no longer evaluated on what it sees.

## Threats and controls

| # | threat | control | tested by |
|---|---|---|---|
| T1 | Poisoned training data (duplication, label flipping, hidden instructions, backdoor triggers) | Poisoning checks; high findings block training until a named acknowledgement | `test_data.py::test_poisoned_example_trips_every_check`, `test_training_gate_needs_validation_then_acknowledgement` |
| T2 | Data changed after validation | Content-addressed versions, re-hashed on load; reports bound by hash | `test_store_is_idempotent_and_detects_tampering` |
| T3 | Held-out contamination inflating scores; choosing the model on the test set | Contamination check between every pair of splits; selection on a validation split, held-out refused; confidence intervals and a paired regression test | `test_contamination_between_evaluation_splits`, `test_sweep_selects_on_validation_and_never_on_heldout`, `test_paired_regression_ignores_noise_and_catches_real_losses` |
| T4 | Base model swapped (moved tag, mirror, cache corruption) | Commit SHA only; SHA-256 per file; re-verified before every use | `test_fetch_refuses_a_file_whose_hash_changed`, `test_verify_detects_tampering_and_extra_files` |
| T5 | Code execution through pickle weights | Safetensors only; pickles (by suffix *or content*) refused; scanner lists imported callables without unpickling | `test_pickle_scan_flags_code_execution_without_running_it`, `test_renamed_pickle_is_still_a_pickle` |
| T6 | Malformed safetensors (out-of-range or overlapping tensors) | Header and layout validated | `test_invalid_safetensors` |
| T7 | Licence violation | Licence table vs intended use; relicensed models denied; conditions need recorded acceptance | `test_licence_table`, `test_conditional_licence_needs_recorded_acceptance` |
| T8 | Memorisation of personal data | Completion-only loss by default; canary extraction and exposure; PII probes vs base model | `test_privacy_gate`, measured in [RESULTS.md](RESULTS.md) |
| T9 | Model easier to attack after fine-tuning; label injection in live requests | Red-team suite vs base model; schema-aware `label_instruction` check on data and on requests (`input_guard` flag/reject) | `test_safety_gate`, `test_attack_detection`, `test_label_instructions`, `test_input_guard_flags_or_rejects_injected_tickets` |
| T10 | Promotion without review | OPA policy: no stage skipping, all gates, independent approvals of the exact manifest | `promotion_test.rego` (15 tests), `test_full_promotion_flow_with_approvals_and_rollback` |
| T11 | Tampered or forged registry artefact | Signed manifest of every file; provenance attestation; re-verified at each transition and before serving | `test_tampering_blocks_promotion_and_is_logged`, `test_forged_manifest_breaks_the_signature` |
| T12 | Hand-edited registry index (production pointer moved without policy) | `verify all` replays stages from the log | `test_hand_edited_production_pointer_is_caught` |
| T13 | Edited, reordered or truncated audit log | Hash chain; audit heads signed into manifests; external anchors | `test_audit.py`, `test_truncated_log_is_caught_by_signed_anchors` |
| T14 | Wrong artefact served (re-tag, edited export) | Gateway verifies registry version, deployment record, export hashes and backend digest; refuses with 503 | `test_gateway_serves_follows_rollback_and_refuses_tampering`, `test_gateway_refuses_a_tampered_export` |
| T15 | Format conversion changes the model | Parity check on held-out at deployment | `test_deploy_refuses_when_the_served_model_does_not_match`; found a real bug, see RESULTS |
| T16 | Silent degradation in service | Drift on inputs and outputs; alerts; retraining proposal (never automatic) | `test_drift_alert_writes_one_proposal_and_never_trains` |
| T17 | Data sent off-site without authority | Content-addressed plan; named, expiring, single-use consent; approvers; personal data acknowledged; re-checked before the first byte | `test_consent_rules`, `test_apply_dry_run_then_execute_once` |
| T18 | Cloud job trains on something else | Job recomputes the dataset version id; base model channel verified against the pin; import checks adapter hashes and plan | `test_import_run_verifies_and_rejoins_the_local_gates`; offline container run |
| T19 | Vulnerable dependencies | Hash-locked installs; pip-audit on both locks; SBOM | CI `supply` job |

## Out of scope and known limits

- **The machine itself.** Someone with root on the host can replace the `lineage`
  binary, the OPA policy or the signing key. Keyless signing in CI and publishing audit
  heads elsewhere limit what they can rewrite unnoticed; they do not prevent it.
- **The audit log is tamper-evident, not tamper-proof**, and lives next to the data it
  describes. Anchoring heads externally (a ticket, a commit, a WORM bucket) is what
  makes rewriting detectable later.
- **Poisoning checks are heuristics.** A clean-label attack that never contradicts the
  rest of its text, or poisoning below the duplicate/conflict thresholds, can pass.
  The backdoor check needs a trigger that occurs in at least three records.
- **The privacy gate measures; it does not prove.** Canaries and probes bound what the
  model memorised of *those* strings; they are evidence, not differential privacy.
- **The red-team suite is small and rule-based** (20 cases). It catches regressions
  between the base and fine-tuned models; it is not a security assessment.
- **Approvals trust `LINEAGE_ACTOR` / the OS user.** Separation of duties is as strong
  as identity on the machine; in a team setting, approvals belong in a system with
  real authentication (the policy only needs the list of approvers).
- **The gateway** is a reference for what to verify before serving, on a standard
  library HTTP server, without authentication or TLS.
- **The AWS path** is tested against botocore's Stubber and an offline container run,
  not against a real account from this repository.
- **Reproducibility is bit-for-bit on the same stack** and was observed across this
  project's server, an offline container and a GitHub-hosted runner, but different CPU
  instruction sets can change floating-point results; it is verified, not assumed.
