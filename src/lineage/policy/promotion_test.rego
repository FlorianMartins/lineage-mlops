package lineage.promotion_test

import rego.v1

import data.lineage.promotion

good_version := {
	"manifest_intact": true,
	"signature_verified": true,
	"provenance_verified": true,
	"eval": {"present": true, "intact": true, "gates": {"quality": true, "privacy": true, "safety": true}},
	"mlbom_present": true,
	"licence_verdict": "allow",
	"licence_accepted": false,
	"manifest_sha256": "abc",
	"trained_by": "alice",
	"registered_by": "alice",
	"approvals": [],
	"was_production": false,
}

settings := {"required_approvals": 1, "separation_of_duties": true}

req(action, from, to, version) := {
	"action": action, "from_stage": from, "to_stage": to,
	"version": version, "actor": "carol", "settings": settings,
}

test_candidate_to_staging_allowed if {
	promotion.allow with input as req("promote", "candidate", "staging", good_version)
}

test_skipping_staging_denied if {
	not promotion.allow with input as req("promote", "candidate", "production", good_version)
}

test_failed_gate_denied if {
	v := json.patch(good_version, [{"op": "replace", "path": "/eval/gates/privacy", "value": false}])
	d := promotion.deny with input as req("promote", "candidate", "staging", v)
	"evaluation gate 'privacy' failed" in d
}

test_missing_gate_denied if {
	v := json.patch(good_version, [{"op": "remove", "path": "/eval/gates/safety"}])
	not promotion.allow with input as req("promote", "candidate", "staging", v)
}

test_unsigned_denied if {
	v := object.union(good_version, {"signature_verified": false})
	not promotion.allow with input as req("promote", "candidate", "staging", v)
}

test_tampered_denied if {
	v := object.union(good_version, {"manifest_intact": false})
	not promotion.allow with input as req("promote", "candidate", "staging", v)
}

test_conditional_licence_needs_acceptance if {
	v := object.union(good_version, {"licence_verdict": "conditional"})
	not promotion.allow with input as req("promote", "candidate", "staging", v)
	accepted := object.union(v, {"licence_accepted": true})
	promotion.allow with input as req("promote", "candidate", "staging", accepted)
}

test_production_needs_an_approval if {
	not promotion.allow with input as req("promote", "staging", "production", good_version)
}

test_self_approval_does_not_count if {
	v := object.union(good_version, {"approvals": [{"actor": "alice", "manifest_sha256": "abc"}]})
	not promotion.allow with input as req("promote", "staging", "production", v)
}

test_approval_of_another_manifest_does_not_count if {
	v := object.union(good_version, {"approvals": [{"actor": "bob", "manifest_sha256": "old"}]})
	not promotion.allow with input as req("promote", "staging", "production", v)
}

test_independent_approval_allows_production if {
	v := object.union(good_version, {"approvals": [{"actor": "bob", "manifest_sha256": "abc"}]})
	promotion.allow with input as req("promote", "staging", "production", v)
}

test_rollback_to_a_former_production_version if {
	v := object.union(good_version, {"was_production": true})
	promotion.allow with input as req("rollback", "archived", "production", v)
}

test_rollback_to_a_never_released_version_denied if {
	not promotion.allow with input as req("rollback", "archived", "production", good_version)
}

test_rolled_back_version_is_terminal if {
	v := object.union(good_version, {"was_production": true})
	not promotion.allow with input as req("rollback", "rolled_back", "production", v)
	not promotion.allow with input as req("promote", "rolled_back", "production", v)
}

test_rollback_still_needs_a_valid_signature if {
	v := object.union(good_version, {"was_production": true, "signature_verified": false})
	not promotion.allow with input as req("rollback", "archived", "production", v)
}
