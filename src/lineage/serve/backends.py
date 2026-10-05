"""Inference backends the gateway talks to.

* ``ollama`` imports the merged safetensors export (``ollama create``) and serves it on
  its HTTP API. Ollama stores the result under a content digest, which deployment
  records and the gateway re-checks before serving.
* ``vllm``   serves the export directory directly through its OpenAI-compatible API
  (``vllm serve <dir>``). Lineage writes the launch command and talks to the server;
  the gateway re-hashes the export directory before trusting it. Not exercised in
  this repository's CI (no GPU, and vLLM's CPU build is heavy).
"""

from __future__ import annotations

import json
import shutil
import subprocess
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from lineage.errors import LineageError


@dataclass
class Completion:
    """One backend answer."""

    text: str
    prompt_tokens: int
    completion_tokens: int


class Backend(Protocol):
    """What the gateway needs from a backend."""

    name: str

    def deploy(self, export_dir: Path, model: str) -> dict[str, Any]:
        """Make ``export_dir`` servable as ``model``; return identifying details."""
        ...

    def digest(self, model: str) -> str | None:
        """Digest of what the backend currently serves under ``model``."""
        ...

    def complete(self, model: str, prompt: str, max_tokens: int) -> Completion:
        """Greedy completion of a raw prompt."""
        ...


def _post(url: str, body: dict[str, Any], timeout: float = 120) -> dict[str, Any]:
    request = urllib.request.Request(  # noqa: S310 - URL comes from the workspace config
        url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            data: dict[str, Any] = json.loads(response.read())
            return data
    except urllib.error.URLError as exc:
        raise LineageError(f"backend at {url} unreachable: {exc}") from exc


def _get(url: str, timeout: float = 30) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310
            data: dict[str, Any] = json.loads(response.read())
            return data
    except urllib.error.URLError as exc:
        raise LineageError(f"backend at {url} unreachable: {exc}") from exc


class Ollama:
    """Ollama on ``url`` (default http://127.0.0.1:11434)."""

    name = "ollama"

    def __init__(self, url: str = "http://127.0.0.1:11434") -> None:
        self.url = url.rstrip("/")

    def deploy(self, export_dir: Path, model: str) -> dict[str, Any]:
        """``ollama create`` from the export's Modelfile."""
        binary = shutil.which("ollama")
        if not binary:
            raise LineageError("ollama is not installed")
        result = subprocess.run(
            [binary, "create", model, "-f", str(export_dir / "Modelfile")],
            capture_output=True,
            text=True,
            timeout=1800,
            cwd=export_dir,
        )
        if result.returncode != 0:
            raise LineageError(f"ollama create failed: {result.stderr.strip()[-300:]}")
        return {"model": model, "digest": self.digest(model)}

    def digest(self, model: str) -> str | None:
        """Digest of ``model`` as listed by the Ollama API."""
        for item in _get(f"{self.url}/api/tags").get("models", []):
            if item.get("name") == model or item.get("model") == model:
                return str(item.get("digest"))
        return None

    def complete(self, model: str, prompt: str, max_tokens: int) -> Completion:
        """Raw prompt (the Modelfile template wraps it), temperature 0."""
        data = _post(
            f"{self.url}/api/generate",
            {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0, "num_predict": max_tokens},
            },
        )
        return Completion(
            str(data.get("response", "")),
            int(data.get("prompt_eval_count", 0)),
            int(data.get("eval_count", 0)),
        )


class VLLM:
    """A vLLM OpenAI-compatible server on ``url``."""

    name = "vllm"

    def __init__(self, url: str = "http://127.0.0.1:8000") -> None:
        self.url = url.rstrip("/")

    def deploy(self, export_dir: Path, model: str) -> dict[str, Any]:
        """Write the launch command; the operator (or a unit file) runs it."""
        command = (
            f"vllm serve {export_dir} --served-model-name {model} "
            f"--dtype float32 --port {self.url.rsplit(':', 1)[-1]}\n"
        )
        (export_dir / "vllm-serve.sh").write_text("#!/bin/sh\nexec " + command)
        (export_dir / "vllm-serve.sh").chmod(0o755)
        return {"model": model, "digest": None, "launch": command.strip()}

    def digest(self, model: str) -> str | None:  # noqa: ARG002 - protocol signature
        """No digest: the gateway re-hashes the export directory vLLM serves from."""
        return None

    def complete(self, model: str, prompt: str, max_tokens: int) -> Completion:
        """OpenAI ``/v1/completions`` with temperature 0."""
        data = _post(
            f"{self.url}/v1/completions",
            {"model": model, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0},
        )
        usage = data.get("usage", {})
        return Completion(
            str(data["choices"][0]["text"]),
            int(usage.get("prompt_tokens", 0)),
            int(usage.get("completion_tokens", 0)),
        )


def make(name: str, url: str | None) -> Backend:
    """Backend by name."""
    if name == "ollama":
        return Ollama(url or "http://127.0.0.1:11434")
    if name == "vllm":
        return VLLM(url or "http://127.0.0.1:8000")
    raise LineageError(f"unknown backend '{name}' (ollama or vllm)")
