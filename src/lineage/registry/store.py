"""A local model registry with stages, signatures and policy-checked transitions.

Layout under ``.lineage/registry/<model>/``::

    index.json                  stage of every version + transition history
    <n>/manifest.json           identity + SHA-256 of every file below (this is signed)
    <n>/manifest.sig.bundle     cosign signature over manifest.json
    <n>/provenance.json         SLSA v1 provenance predicate
    <n>/provenance.sig.bundle   cosign attestation of the provenance over manifest.json
    <n>/adapter/...             the LoRA adapter (safetensors)
    <n>/run.json, eval.json     copies of the run record and evaluation report
    <n>/mlbom.cdx.json          CycloneDX ML-BOM
    <n>/MODEL_CARD.md           human summary

Stages: ``candidate`` -> ``staging`` -> ``production``. Promoting a new version to
production archives the previous one; ``rollback`` brings the last archived production
version back. Every transition re-verifies hashes and signatures, asks the OPA policy,
and is written to the audit log whether it was allowed or denied.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from datetime import UTC, datetime
from importlib import resources
from pathlib import Path
from typing import Any

from lineage.audit import AuditLog
from lineage.errors import IntegrityError, LineageError, PolicyDenied
from lineage.hashing import sha256_file, sha256_json
from lineage.registry.signing import Signer
from lineage.workspace import Workspace, current_actor

STAGES = ("candidate", "staging", "production", "archived", "rolled_back")
SIGNED = {"manifest.json", "manifest.sig.bundle", "provenance.sig.bundle"}


def _now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_ref(ref: str) -> tuple[str, int | None]:
    """``name:3`` -> ("name", 3); ``name`` -> ("name", None)."""
    name, _, number = ref.partition(":")
    if number and not number.isdigit():
        raise LineageError(f"'{ref}': version must be a number (name:3)")
    return name, int(number) if number else None


class Registry:
    """Operations on one workspace's registry."""

    def __init__(self, ws: Workspace) -> None:
        self.ws = ws
        settings = ws.section("registry")
        self.model_name = str(settings.get("model_name", ws.section("task").get("name", "model")))
        self.required_approvals = int(settings.get("required_approvals", 1))
        self.separation_of_duties = bool(settings.get("separation_of_duties", True))
        policy_dir = settings.get("policy_dir")
        self.policy_dir = ws.resolve(str(policy_dir)) if policy_dir else None
        self.log = AuditLog(ws.audit_log)

    # -- paths and index -----------------------------------------------------------
    def root(self, name: str | None = None) -> Path:
        """Directory of a model."""
        return self.ws.registry / (name or self.model_name)

    def path(self, number: int, name: str | None = None) -> Path:
        """Directory of a version."""
        return self.root(name) / str(number)

    def index(self, name: str | None = None) -> dict[str, Any]:
        """The model's index (empty if nothing registered yet)."""
        path = self.root(name) / "index.json"
        if not path.exists():
            return {"model": name or self.model_name, "versions": {}, "history": []}
        data: dict[str, Any] = json.loads(path.read_text())
        return data

    def _save_index(self, index: dict[str, Any]) -> None:
        path = self.root(index["model"]) / "index.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(index, indent=2) + "\n")
        tmp.replace(path)

    def versions(self, name: str | None = None) -> list[dict[str, Any]]:
        """All versions with their stage, oldest first."""
        index = self.index(name)
        return [
            {"version": int(n), **entry}
            for n, entry in sorted(index["versions"].items(), key=lambda kv: int(kv[0]))
        ]

    def production(self, name: str | None = None) -> dict[str, Any] | None:
        """The version currently in production, if any."""
        for entry in self.versions(name):
            if entry["stage"] == "production":
                return entry
        return None

    def resolve(self, ref: str) -> tuple[str, int]:
        """``name:3``, ``name`` (= production) or ``3`` (= this model, version 3)."""
        if ref.isdigit():
            return self.model_name, int(ref)
        name, number = parse_ref(ref)
        if number is None:
            current = self.production(name)
            if current is None:
                raise LineageError(f"{name} has no production version")
            number = current["version"]
        if str(number) not in self.index(name)["versions"]:
            raise LineageError(f"{name}:{number} is not registered")
        return name, number

    # -- evidence ---------------------------------------------------------------------
    def manifest(self, number: int, name: str | None = None) -> dict[str, Any]:
        """The version's manifest."""
        data: dict[str, Any] = json.loads((self.path(number, name) / "manifest.json").read_text())
        return data

    def eval_report(self, entry: dict[str, Any], name: str | None = None) -> dict[str, Any]:
        """The stored evaluation report of a version."""
        data: dict[str, Any] = json.loads(
            (self.path(entry["version"], name) / "eval.json").read_text()
        )
        return data

    def check_files(self, number: int, name: str | None = None) -> list[str]:
        """Re-hash every file listed in the manifest; return problems (empty = intact)."""
        base = self.path(number, name)
        manifest = self.manifest(number, name)
        problems = []
        for rel, digest in manifest["files"].items():
            path = base / rel
            if not path.exists():
                problems.append(f"{rel} is missing")
            elif sha256_file(path) != digest:
                problems.append(f"{rel} does not match the manifest")
        present = {str(p.relative_to(base)) for p in base.rglob("*") if p.is_file()} - SIGNED
        extra = present - set(manifest["files"])
        problems += [f"{rel} is not in the manifest" for rel in sorted(extra)]
        return problems

    def verify(self, number: int, name: str | None = None) -> dict[str, Any]:
        """Files, signature and provenance of one version, as booleans with reasons."""
        base = self.path(number, name)
        problems = self.check_files(number, name)
        result: dict[str, Any] = {"manifest_intact": not problems, "problems": problems}
        signer = Signer.from_workspace(self.ws)
        for key, method, bundle in (
            ("signature_verified", signer.verify, "manifest.sig.bundle"),
            ("provenance_verified", signer.verify_attestation, "provenance.sig.bundle"),
        ):
            try:
                method(base / "manifest.json", base / bundle)
                result[key] = True
            except PolicyDenied as exc:
                result[key] = False
                result["problems"].append(str(exc))
        result["ok"] = (
            result["manifest_intact"]
            and result["signature_verified"]
            and result["provenance_verified"]
        )
        return result

    # -- policy -------------------------------------------------------------------------
    def decide(self, request: dict[str, Any]) -> dict[str, Any]:
        """Ask OPA. No OPA, no decision: fail closed."""
        opa = shutil.which("opa")
        if not opa:
            raise PolicyDenied("opa is not installed: promotions fail closed")
        if self.policy_dir:
            policy = self.policy_dir
            return self._eval(opa, policy, request)
        with resources.as_file(resources.files("lineage") / "policy") as policy:
            return self._eval(opa, Path(policy), request)

    @staticmethod
    def _eval(opa: str, policy: Path, request: dict[str, Any]) -> dict[str, Any]:
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
            json.dump(request, handle)
            input_path = handle.name
        try:
            result = subprocess.run(
                [
                    opa,
                    "eval",
                    "--format",
                    "json",
                    "-d",
                    str(policy),
                    "-i",
                    input_path,
                    "data.lineage.promotion.decision",
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )
        finally:
            Path(input_path).unlink(missing_ok=True)
        if result.returncode != 0:
            raise PolicyDenied(f"policy evaluation failed: {result.stderr.strip()[:300]}")
        try:
            value = json.loads(result.stdout)["result"][0]["expressions"][0]["value"]
        except (KeyError, IndexError, ValueError) as exc:
            raise PolicyDenied("policy returned no decision") from exc
        return {
            "allow": bool(value["allow"]),
            "deny": sorted(value["deny"]),
            "policy_sha256": sha256_json(_policy_files(policy)),
        }


def _policy_files(policy: Path) -> dict[str, str]:
    return {p.name: sha256_file(p) for p in sorted(policy.glob("*.rego"))}


def write_json(path: Path, data: Any) -> None:
    """Write pretty JSON."""
    path.write_text(json.dumps(data, indent=2) + "\n")


def ensure_integrity(problems: list[str], what: str) -> None:
    """Raise when a re-hash found anything."""
    if problems:
        raise IntegrityError(f"{what}: " + "; ".join(problems))


__all__ = ["STAGES", "Registry", "current_actor", "ensure_integrity", "parse_ref", "write_json"]
