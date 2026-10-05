"""Exceptions that carry a decision, not just a failure."""

from __future__ import annotations


class LineageError(Exception):
    """Base class: the CLI prints the message and exits with code 2."""


class IntegrityError(LineageError):
    """Something on disk no longer matches the digest it was recorded with."""


class PolicyDenied(LineageError):
    """A gate, a policy or a consent check refused the action.

    The CLI exits with code 1 for this one: the system worked, the answer was no.
    """

    def __init__(self, message: str, reasons: list[str] | None = None) -> None:
        super().__init__(message)
        self.reasons = reasons or []
