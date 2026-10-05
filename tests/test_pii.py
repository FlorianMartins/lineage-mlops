import pytest

from lineage.data import pii


@pytest.mark.parametrize(
    ("text", "kind"),
    [
        ("mail alice.martin@example.com now", "email"),
        ("card 4111 1111 1111 1111 please", "payment_card"),
        ("IBAN GB82 WEST 1234 5698 7654 32", "iban"),
        ("call +44 20 7946 0958 today", "phone"),
        ("server 10.0.12.7 is down", "ipv4"),
        ("ssn 123-45-6789", "us_ssn"),
        ("key AKIAIOSFODNN7EXAMPLE leaked", "aws_access_key"),
        ("token ghp_" + "a" * 36, "github_token"),
        ("sk-" + "x" * 30, "api_key"),
    ],
)
def test_detects(text, kind):
    assert [m.kind for m in pii.scan(text)] == [kind]


@pytest.mark.parametrize(
    "text",
    [
        "card 4111 1111 1111 1112 fails luhn",
        "IBAN GB00 WEST 1234 5698 7654 32 bad checksum",
        "release 2026 of version 1.2.3",
        "room 101 on floor 3",
        "ticket number 20261005",
    ],
)
def test_ignores_lookalikes(text):
    assert pii.scan(text) == []


def test_masking_never_returns_the_value():
    match = pii.scan("write to alice.martin@example.com")[0]
    assert match.masked != match.value
    assert "martin" not in match.masked
    assert pii.mask("abc") == "***"


def test_redact():
    assert pii.redact("ping bob@corp.io or 10.1.2.3") == "ping [EMAIL] or [IPV4]"
