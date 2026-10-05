# Security policy

## Reporting a vulnerability

Please report security issues privately through
[GitHub security advisories](https://github.com/FlorianMartins/lineage-mlops/security/advisories/new)
rather than a public issue. You can expect an acknowledgement within a few days.

## Scope

Lineage is a reference implementation of a secured MLOps lifecycle. Reports are
especially welcome when they show a way to:

- get a model into production without passing the gates, the policy or an approval;
- make the audit log, a signature or a manifest verify after tampering;
- make the gateway serve something other than the verified production version;
- send data off the machine without a valid consent;
- get a pickle-based file loaded, or a base model that does not match its pin.

The threat model in [docs/THREAT_MODEL.md](docs/THREAT_MODEL.md) lists what is in and
out of scope, and the known limits.
