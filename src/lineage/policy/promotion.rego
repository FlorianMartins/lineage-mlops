# Promotion policy for the Lineage model registry.
#
# The registry builds `input` from evidence it has just re-verified (hashes, signature,
# evaluation report) and asks this policy for `decision`. Python never decides on its
# own: if OPA is missing, promotions fail closed.
#
# Changing a rule here changes what can reach production, so this file is reviewed
# like code and tested with `opa test` (promotion_test.rego).
package lineage.promotion

import rego.v1

default allow := false

allow if count(deny) == 0

decision := {"allow": allow, "deny": deny}

transitions := {
	"promote": {["candidate", "staging"], ["staging", "production"]},
	# A version that was rolled back is terminal: rolling back twice must not bring
	# back the release that was pulled for a reason.
	"rollback": {["archived", "production"]},
}

v := input.version

# --- every action -------------------------------------------------------------

deny contains msg if {
	not [input.from_stage, input.to_stage] in transitions[input.action]
	msg := sprintf("%s from %s to %s is not an allowed transition", [input.action, input.from_stage, input.to_stage])
}

deny contains "the version's files no longer match its manifest" if not v.manifest_intact

deny contains "the manifest signature does not verify" if not v.signature_verified

deny contains "the provenance attestation does not verify" if not v.provenance_verified

# --- promotions ---------------------------------------------------------------

deny contains "no evaluation report" if {
	input.action == "promote"
	not v.eval.present
}

deny contains "the evaluation report is not intact or not about this adapter" if {
	input.action == "promote"
	v.eval.present
	not v.eval.intact
}

deny contains msg if {
	input.action == "promote"
	some gate, passed in v.eval.gates
	not passed
	msg := sprintf("evaluation gate '%s' failed", [gate])
}

deny contains "required gate missing from the evaluation (quality, privacy, safety)" if {
	input.action == "promote"
	some gate in {"quality", "privacy", "safety"}
	not gate in object.keys(object.get(v.eval, "gates", {}))
}

deny contains "no ML-BOM" if {
	input.action == "promote"
	not v.mlbom_present
}

deny contains "the base model licence does not permit the intended use" if {
	input.action == "promote"
	v.licence_verdict == "deny"
}

deny contains "the base model licence has conditions nobody accepted" if {
	input.action == "promote"
	v.licence_verdict == "conditional"
	not v.licence_accepted
}

# --- production needs people --------------------------------------------------

# Approvals count only if they are about this exact manifest and, with separation of
# duties, if the approver neither trained nor registered the model.
valid_approvers contains a.actor if {
	some a in v.approvals
	a.manifest_sha256 == v.manifest_sha256
	not conflicted(a.actor)
}

conflicted(actor) if {
	input.settings.separation_of_duties
	actor in {v.trained_by, v.registered_by}
}

deny contains msg if {
	input.action == "promote"
	input.to_stage == "production"
	count(valid_approvers) < input.settings.required_approvals
	msg := sprintf(
		"production needs %d independent approval(s) of this manifest, found %d",
		[input.settings.required_approvals, count(valid_approvers)],
	)
}

# --- rollback -----------------------------------------------------------------

deny contains "only a version that was in production before can be rolled back to" if {
	input.action == "rollback"
	not v.was_production
}
