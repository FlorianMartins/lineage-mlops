"""Signatures and provenance with Sigstore cosign.

Two modes, chosen in ``[signing]``:

* ``key``      a cosign key pair (``lineage signing init``). Works offline; nothing is
               published to a transparency log. Fit for a laptop or an air-gapped site.
* ``keyless``  Sigstore keyless: an OIDC identity (in CI, the GitHub Actions workflow)
               gets a short-lived certificate from Fulcio, and the signature is
               recorded in the public Rekor log. Verification pins the identity and
               issuer, so "signed" means "signed by *that* workflow".

What is signed is the version's ``manifest.json``, which lists the SHA-256 of every
file in the version; verifying the signature and re-hashing the files covers them all.
The SLSA provenance is a separate in-toto attestation over the same manifest.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lineage.errors import LineageError, PolicyDenied
from lineage.workspace import Workspace

PREDICATE_TYPE = "slsaprovenance1"


def cosign() -> str:
    """Path to the cosign binary, or a refusal (signing fails closed)."""
    path = shutil.which("cosign")
    if not path:
        raise PolicyDenied("cosign is not installed: nothing can be signed or verified")
    return path


@dataclass(frozen=True)
class Signer:
    """Signs and verifies blobs and attestations."""

    mode: str
    key: Path | None = None
    public_key: Path | None = None
    identity: str | None = None
    issuer: str | None = None

    @classmethod
    def from_workspace(cls, ws: Workspace) -> Signer:
        """Read ``[signing]`` (default: key mode with keys under ``.lineage/keys``).

        ``LINEAGE_SIGNING_MODE``, ``LINEAGE_SIGNING_IDENTITY`` and
        ``LINEAGE_SIGNING_ISSUER`` override the file, so the same workspace signs with a
        local key on a laptop and keylessly with the workflow identity in CI.
        """
        section = dict(ws.section("signing"))
        for key in ("mode", "identity", "issuer"):
            value = os.environ.get(f"LINEAGE_SIGNING_{key.upper()}")
            if value:
                section[key] = value
        mode = str(section.get("mode", "key"))
        if mode == "key":
            return cls(
                mode,
                key=ws.resolve(str(section.get("key", ".lineage/keys/cosign.key"))),
                public_key=ws.resolve(str(section.get("public_key", ".lineage/keys/cosign.pub"))),
            )
        if mode == "keyless":
            identity = section.get("identity")
            issuer = section.get("issuer", "https://token.actions.githubusercontent.com")
            if not identity:
                raise LineageError("[signing] keyless mode needs 'identity' (a regexp)")
            return cls(mode, identity=str(identity), issuer=str(issuer))
        raise LineageError(f"[signing].mode must be 'key' or 'keyless', not '{mode}'")

    # -- low level --------------------------------------------------------------
    def _run(self, args: list[str]) -> None:
        env = dict(os.environ)
        env.setdefault("COSIGN_PASSWORD", "")
        result = subprocess.run(
            [cosign(), *args], capture_output=True, text=True, env=env, timeout=300
        )
        if result.returncode != 0:
            message = (result.stderr or result.stdout).strip().splitlines()
            raise PolicyDenied(f"cosign {args[0]} failed: {message[-1] if message else '?'}")

    def _sign_flags(self) -> list[str]:
        if self.mode == "key":
            assert self.key is not None
            if not self.key.exists():
                raise PolicyDenied(f"signing key {self.key} not found (run `lineage signing init`)")
            return [
                "--key",
                str(self.key),
                "--use-signing-config=false",
                "--tlog-upload=false",
                "--yes",
            ]
        return ["--yes"]

    def _verify_flags(self) -> list[str]:
        if self.mode == "key":
            assert self.public_key is not None
            return ["--key", str(self.public_key), "--insecure-ignore-tlog"]
        assert self.identity and self.issuer
        return [
            "--certificate-identity-regexp",
            self.identity,
            "--certificate-oidc-issuer",
            self.issuer,
        ]

    # -- API ----------------------------------------------------------------------
    def sign(self, blob: Path, bundle: Path) -> None:
        """Sign ``blob``; write a Sigstore bundle."""
        self._run(["sign-blob", *self._sign_flags(), "--bundle", str(bundle), str(blob)])

    def attest(self, blob: Path, predicate: Path, bundle: Path) -> None:
        """Attest SLSA provenance about ``blob``."""
        self._run(
            [
                "attest-blob",
                *self._sign_flags(),
                "--predicate",
                str(predicate),
                "--type",
                PREDICATE_TYPE,
                "--bundle",
                str(bundle),
                str(blob),
            ]
        )

    def verify(self, blob: Path, bundle: Path) -> None:
        """Raise unless ``bundle`` is a valid signature over ``blob``."""
        if not bundle.exists():
            raise PolicyDenied(f"{blob.parent.name}: no signature")
        self._run(["verify-blob", *self._verify_flags(), "--bundle", str(bundle), str(blob)])

    def verify_attestation(self, blob: Path, bundle: Path) -> None:
        """Raise unless ``bundle`` is a valid provenance attestation over ``blob``."""
        if not bundle.exists():
            raise PolicyDenied(f"{blob.parent.name}: no provenance attestation")
        self._run(
            [
                "verify-blob-attestation",
                *self._verify_flags(),
                "--type",
                PREDICATE_TYPE,
                "--bundle",
                str(bundle),
                str(blob),
            ]
        )

    def describe(self) -> dict[str, Any]:
        """What verification will require (recorded with each signature)."""
        if self.mode == "key":
            from lineage.hashing import sha256_file

            assert self.public_key is not None
            digest = sha256_file(self.public_key) if self.public_key.exists() else None
            return {"mode": "key", "public_key_sha256": digest}
        return {"mode": "keyless", "identity": self.identity, "issuer": self.issuer}


def init_keys(ws: Workspace) -> Path:
    """Generate a cosign key pair under ``.lineage/keys`` (password from COSIGN_PASSWORD)."""
    target = ws.state / "keys"
    if (target / "cosign.key").exists():
        raise LineageError(f"{target / 'cosign.key'} already exists")
    target.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.setdefault("COSIGN_PASSWORD", "")
    result = subprocess.run(
        [cosign(), "generate-key-pair"],
        cwd=target,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0:
        raise LineageError(f"cosign generate-key-pair failed: {result.stderr.strip()}")
    (target / "cosign.key").chmod(0o600)
    return target
