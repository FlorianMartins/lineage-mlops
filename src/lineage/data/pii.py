"""Personal-data scanner.

Regex alone over-reports, so numeric identifiers are confirmed with their checksum
(Luhn for payment cards, ISO 7064 mod-97 for IBANs). Matches are always reported
*masked*: a PII report must not become a second copy of the PII.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class PiiMatch:
    """One detected item."""

    kind: str
    value: str
    start: int
    end: int

    @property
    def masked(self) -> str:
        """Keep the shape, hide the content."""
        return mask(self.value)


def mask(value: str) -> str:
    """``alice@example.com`` -> ``a****@e**********``-style masking."""
    if len(value) <= 4:
        return "*" * len(value)
    return value[:2] + "*" * (len(value) - 4) + value[-2:]


def _luhn_ok(digits: str) -> bool:
    total = 0
    for index, char in enumerate(reversed(digits)):
        n = int(char)
        if index % 2 == 1:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0


def _iban_ok(raw: str) -> bool:
    iban = raw.replace(" ", "").upper()
    if len(iban) < 15 or len(iban) > 34:
        return False
    rearranged = iban[4:] + iban[:4]
    numeric = "".join(str(int(ch, 36)) for ch in rearranged)
    return int(numeric) % 97 == 1


_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("email", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
    ("iban", re.compile(r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){2,7}(?: ?[A-Z0-9]{1,4})?\b")),
    ("payment_card", re.compile(r"\b(?:\d[ -]?){12,18}\d\b")),
    ("us_ssn", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    # International (+CC) or national trunk-prefix (0...) numbers only: a bare run of
    # digits is far more often an order number than a phone number.
    (
        "phone",
        re.compile(r"(?<![\w+])(?:\+\d{1,3}|\(?0\d{1,4}\)?)(?:[ .-]?\d{2,4}){2,5}(?!\w)"),
    ),
    ("ipv4", re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b")),
    ("aws_access_key", re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")),
    ("github_token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}\b")),
    ("api_key", re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b")),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b")),
]


def scan(text: str) -> list[PiiMatch]:
    """Return every PII item in ``text``; overlapping weaker matches are dropped."""
    found: list[PiiMatch] = []
    taken: list[tuple[int, int]] = []

    def free(start: int, end: int) -> bool:
        return all(end <= s or start >= e for s, e in taken)

    for kind, pattern in _PATTERNS:
        for match in pattern.finditer(text):
            value = match.group(0)
            start, end = match.span()
            if not free(start, end):
                continue
            if not _confirmed(kind, value):
                continue
            found.append(PiiMatch(kind, value, start, end))
            taken.append((start, end))
    return sorted(found, key=lambda m: m.start)


def _confirmed(kind: str, value: str) -> bool:
    digits = re.sub(r"\D", "", value)
    if kind == "payment_card":
        return 13 <= len(digits) <= 19 and _luhn_ok(digits)
    if kind == "iban":
        return _iban_ok(value)
    if kind == "phone":
        # E.164 allows at most 15 digits; fewer than 9 is an extension or a year.
        return 9 <= len(digits) <= 15
    if kind == "ipv4":
        # Version strings like 1.2.3.4 are rare in tickets; private ranges still count,
        # they identify a machine inside an organisation.
        return True
    return True


def redact(text: str) -> str:
    """Replace every match with ``[KIND]`` (used for display and for drift samples)."""
    out = []
    cursor = 0
    for match in scan(text):
        out.append(text[cursor : match.start])
        out.append(f"[{match.kind.upper()}]")
        cursor = match.end
    out.append(text[cursor:])
    return "".join(out)
