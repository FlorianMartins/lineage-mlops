"""Base-model licence check against the intended use.

The decision is deliberately conservative: a licence the table does not know is a
``deny`` until someone adds it, because "probably fine" is how unlicensed models end
up in production. ``conditional`` means the use is permitted under terms a human must
accept (an acceptable-use policy, a user cap, attribution); training proceeds only once
that acceptance is recorded in the audit log.

This is an engineering control, not legal advice: the table encodes a reading of each
licence that legal should review once and own.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from lineage.errors import LineageError

USES = ("research", "internal", "commercial", "redistribution")


@dataclass(frozen=True)
class LicenceTerms:
    """What a licence permits, in the four uses Lineage reasons about.

    ``obligations`` are mechanical duties (keep a notice, attribute): recorded, not
    blocking. ``conditions`` are terms someone must read and accept on behalf of the
    organisation (an acceptable-use policy, a user cap): blocking until accepted.
    """

    spdx: str
    permits: frozenset[str]
    obligations: tuple[str, ...] = ()
    conditions: tuple[str, ...] = ()


_TABLE: dict[str, LicenceTerms] = {}


def _add(
    spdx: str,
    permits: set[str],
    *,
    obligations: tuple[str, ...] = (),
    conditions: tuple[str, ...] = (),
    aliases: tuple[str, ...] = (),
) -> None:
    terms = LicenceTerms(spdx, frozenset(permits), obligations, conditions)
    for key in (spdx, *aliases):
        _TABLE[key.lower()] = terms


# Canonical SPDX identifiers (hubs report them lower-cased). Licences without an SPDX
# id (Llama, Gemma, OpenRAIL variants) are reported by name in the ML-BOM.
SPDX = {
    "apache-2.0": "Apache-2.0",
    "mit": "MIT",
    "bsd-3-clause": "BSD-3-Clause",
    "cc-by-4.0": "CC-BY-4.0",
    "cc-by-nc-4.0": "CC-BY-NC-4.0",
    "cc-by-nc-sa-4.0": "CC-BY-NC-SA-4.0",
    "cc-by-nc-nd-4.0": "CC-BY-NC-ND-4.0",
}


def cyclonedx_licence(key: str | None) -> list[dict[str, object]]:
    """The ``licenses`` entry of a CycloneDX component for a hub licence key."""
    if not key or key == "unknown":
        return [{"expression": "NOASSERTION"}]
    spdx = SPDX.get(key.lower())
    return [{"license": {"id": spdx}}] if spdx else [{"license": {"name": key}}]


_ALL = set(USES)
_NOTICE = ("keep the licence text and notices when redistributing",)
_add("apache-2.0", _ALL, obligations=_NOTICE)
_add("mit", _ALL, obligations=_NOTICE)
_add("bsd-3-clause", _ALL, obligations=_NOTICE)
_add("cc-by-4.0", _ALL, obligations=("attribute the original authors",))
_add(
    "openrail",
    _ALL,
    conditions=("respect the use restrictions in the licence attachment",),
    aliases=("openrail++", "creativeml-openrail-m", "bigscience-openrail-m"),
)
_add(
    "llama3.1",
    _ALL,
    obligations=("display 'Built with Llama'; derived model names must include 'Llama'",),
    conditions=(
        "accept the Llama acceptable-use policy",
        "a separate licence is needed above 700M monthly active users",
    ),
    aliases=("llama3", "llama3.2", "llama3.3", "llama2"),
)
_add("gemma", _ALL, conditions=("accept the Gemma prohibited-use policy and pass it on to users",))
_add(
    "cc-by-nc-4.0",
    {"research", "internal"},
    obligations=("attribute the original authors",),
    aliases=("cc-by-nc-sa-4.0", "cc-by-nc-nd-4.0"),
)
_add("research-only", {"research"})


@dataclass
class LicenceDecision:
    """Outcome of a check."""

    licence: str
    use: str
    verdict: str  # allow | conditional | deny
    reasons: list[str] = field(default_factory=list)
    conditions: list[str] = field(default_factory=list)
    obligations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Report form."""
        return {
            "licence": self.licence,
            "use": self.use,
            "verdict": self.verdict,
            "reasons": self.reasons,
            "conditions": self.conditions,
            "obligations": self.obligations,
        }


def check(licence: str | None, use: str, *, declared: str | None = None) -> LicenceDecision:
    """Decide whether ``use`` is permitted by ``licence``.

    ``declared`` is the licence written in the pin. If the hub now reports a different
    one, the model was relicensed after it was reviewed: deny until re-reviewed.
    """
    if use not in USES:
        raise LineageError(f"intended use must be one of {USES}, got '{use}'")
    key = (licence or "").strip().lower()
    decision = LicenceDecision(licence=key or "unknown", use=use, verdict="deny")
    if declared and declared.strip().lower() != key:
        decision.reasons.append(
            f"licence reported as '{key or 'none'}' but pinned as '{declared}': "
            "re-review before training"
        )
        return decision
    terms = _TABLE.get(key)
    if terms is None:
        decision.reasons.append(f"licence '{key or 'none'}' is not in the reviewed table")
        return decision
    if use not in terms.permits:
        decision.reasons.append(f"'{terms.spdx}' does not permit {use} use")
        return decision
    decision.conditions = list(terms.conditions)
    decision.obligations = list(terms.obligations)
    decision.verdict = "conditional" if terms.conditions else "allow"
    decision.reasons.append(f"'{terms.spdx}' permits {use} use")
    return decision
