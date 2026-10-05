# Runbooks

What to do when Lineage says no. Each section starts from the message or alert you see.

## Verification failure

*Alert `LineageVerificationFailure`; gateway answers 503; `serve.refused` in the log.*

The gateway refused to serve because something it checks did not match: the registry
version (files, signature, provenance), the deployment record, the export directory, or
the digest the backend reports for the model tag.

1. `curl -s localhost:8765/healthz` shows the reason in `detail`.
2. `lineage registry verify <model>` and `lineage verify all` tell you which artefact
   disagrees.
3. If the backend digest differs, someone re-created or re-tagged the model in Ollama.
   Do **not** "fix" the record. Find out who and why (`ollama list`, host logs), then
   redeploy from the registry: `lineage deploy run <model>:<n>`.
4. If registry files or signatures fail, treat it as an incident: the version on disk is
   not the one that was approved. Roll back (`lineage registry rollback --reason ...`)
   to restore service with a verified version, preserve the workspace for analysis.

## Drift

*Alert `LineageDriftDetected`; a file in `.lineage/proposals/`.*

1. Read the proposal: which signals crossed which thresholds, on how many requests.
2. Decide whether the traffic changed (new kind of tickets, new language, a new product)
   or something upstream broke (a client sending empty or truncated text). Input-only
   drift with stable outputs is often upstream; output drift means answers changed.
3. If the world changed: sample and label recent tickets (the gateway keeps no text, so
   this comes from the ticketing system), then follow the proposal's steps: ingest a new
   version, validate, plant canaries, train, evaluate, register, promote with approval.
4. If the model is now wrong in a way that matters, roll back first, investigate after.
5. Record the decision: edit the proposal's `status` (`accepted`, `rejected`) and reason.

Lineage never retrains automatically: a drift alert can be caused by an attacker
flooding the endpoint, and an automatic retrain would hand them the training data.

## A gate failed

*`lineage eval run` exits 1.*

- **quality, "below the RAG baseline"**: retrieval alone does as well; fine-tuning is
  not worth maintaining. Ship RAG, or improve the data, not the hyperparameters.
- **quality, "regression vs production"**: the candidate is worse than what runs now.
- **privacy, canary exposure**: the model memorised the planted secrets, so it can
  memorise real personal data. Check `loss_on` (use `completion`), duplicates in the
  data, and the number of epochs. Do not raise the threshold to make it pass.
- **safety**: look at `results.redteam.finetuned.successes` in the report; each entry
  says which case succeeded and why.

## Promotion denied

The message lists every reason. The usual ones: a stage skipped, a failed gate, an
approval missing or not independent (the approver trained or registered the model), an
approval of an older manifest, a conditional licence not accepted. Fix the cause; the
policy is in [`src/lineage/policy/promotion.rego`](../src/lineage/policy/promotion.rego)
and changes to it are code changes, reviewed and tested.

## Rollback

```bash
lineage registry rollback --reason "what broke, ticket number"
curl -s localhost:8765/healthz      # the gateway follows within one request
```

The previous production version must still be deployed on the backend (deployments are
kept per version); if it is not, run `lineage deploy run <model>:<n>` first.

## Audit log broken

*`lineage audit verify` or `verify all` reports a broken chain or a missing anchor.*

The log was edited, truncated or restored from an old backup. Stop promotions, keep a
copy of the current file, and compare with the last published head (`--anchor`). The
entries before the first break are still trustworthy; everything after must be
re-established from the artefacts and their signatures.
