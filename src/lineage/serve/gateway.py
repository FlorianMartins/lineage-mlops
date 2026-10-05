"""The serving gateway: verify, then serve, then watch.

``lineage serve`` answers ``POST /v1/triage`` by calling the backend, and only ever
for the version the registry currently names as production. Before it serves a
version, and again whenever the production pointer moves (a promotion or a rollback),
it checks:

1. the registry version: file hashes, cosign signature, provenance attestation;
2. the deployment record: the hash the audit log recorded for it;
3. the artefact actually served: the export directory's hashes and, for Ollama, the
   digest Ollama reports for the model tag.

If any check fails it refuses to serve (HTTP 503) rather than fall back to something
unverified. Rolling back in the registry is therefore all it takes to roll back
serving.

``GET /metrics`` exposes Prometheus metrics (latency, errors, tokens, invalid answers,
drift scores); ``GET /healthz`` says which version is live. With the ``otel`` extra and
``OTEL_EXPORTER_OTLP_ENDPOINT`` set, each request is also an OpenTelemetry span.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from lineage.audit import AuditLog
from lineage.data import service as data_service
from lineage.errors import IntegrityError, LineageError, PolicyDenied
from lineage.hashing import sha256_file
from lineage.registry.store import Registry
from lineage.serve import backends, deploy, drift
from lineage.workspace import Workspace

log = logging.getLogger("lineage.gateway")
MAX_BODY = 16 * 1024


class Metrics:
    """Prometheus collectors on a private registry (one gateway per process)."""

    def __init__(self) -> None:
        from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

        self.registry = CollectorRegistry()
        self.requests = Counter(
            "lineage_requests_total", "Triage requests", ["status"], registry=self.registry
        )
        self.latency = Histogram(
            "lineage_request_seconds",
            "End-to-end request latency",
            buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10),
            registry=self.registry,
        )
        self.tokens = Counter(
            "lineage_tokens_total",
            "Tokens processed by the backend",
            ["kind"],
            registry=self.registry,
        )
        self.invalid = Counter(
            "lineage_invalid_answers_total",
            "Backend answers that are not valid triage answers",
            registry=self.registry,
        )
        self.drift = Gauge(
            "lineage_drift_score",
            "Drift score against the evaluation set",
            ["signal"],
            registry=self.registry,
        )
        self.alerts = Counter(
            "lineage_drift_alerts_total", "Drift alerts raised", ["signal"], registry=self.registry
        )
        self.live = Gauge(
            "lineage_model_info",
            "Version being served (value 1)",
            ["model", "version", "backend"],
            registry=self.registry,
        )
        self.verify_failures = Counter(
            "lineage_verification_failures_total",
            "Refusals to serve because verification failed",
            registry=self.registry,
        )

    def render(self) -> bytes:
        """Text exposition format."""
        from prometheus_client import generate_latest

        payload: bytes = generate_latest(self.registry)
        return payload


def _tracer() -> Any:
    if not os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
        return None
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError:
        log.warning("OTEL_EXPORTER_OTLP_ENDPOINT is set but the [otel] extra is not installed")
        return None
    provider = TracerProvider(resource=Resource.create({"service.name": "lineage-gateway"}))
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(provider)
    return trace.get_tracer("lineage.gateway")


class Live:
    """The verified version being served, re-resolved when production moves."""

    def __init__(self, ws: Workspace, backend: backends.Backend, metrics: Metrics) -> None:
        self.ws = ws
        self.backend = backend
        self.metrics = metrics
        self.registry = Registry(ws)
        self.lock = threading.Lock()
        self.index_mtime: float | None = None
        self.current: dict[str, Any] | None = None
        self.error: str | None = None
        self.monitor: drift.DriftMonitor | None = None
        self.task = data_service.task_of(ws)
        self.verified_at = 0.0
        # Re-verify periodically too: someone can re-tag the backend model without
        # touching the registry.
        self.reverify_seconds = float(ws.section("serve").get("reverify_seconds", 60))

    def _index_path(self) -> Path:
        return self.registry.root() / "index.json"

    def refresh(self) -> None:
        """Re-verify if the registry index changed since the last check."""
        path = self._index_path()
        mtime = path.stat().st_mtime if path.exists() else None
        fresh = time.monotonic() - self.verified_at < self.reverify_seconds
        if mtime == self.index_mtime and fresh and (self.current or self.error):
            return
        with self.lock:
            self.index_mtime = mtime
            self.verified_at = time.monotonic()
            previous = self.current
            try:
                self.current = self._verify(previous)
                self.error = None
            except (LineageError, OSError) as exc:
                self.current = None
                self.error = str(exc)
                self.metrics.verify_failures.inc()
                log.error("refusing to serve: %s", exc)
                AuditLog(self.ws.audit_log).append(
                    "serve.refused", payload={"reason": str(exc)[:500]}
                )

    def _verify(self, previous: dict[str, Any] | None) -> dict[str, Any]:
        production = self.registry.production()
        if production is None:
            raise LineageError("no version in production")
        name, number = self.registry.model_name, int(production["version"])
        deploy.verified(self.ws, f"{name}:{number}")
        record = deploy.load_deployment(self.ws, name, number, self.backend.name)
        path = deploy.deployments(self.ws) / f"{name}-v{number}-{self.backend.name}.json"
        logged = [
            e.payload.get("deployment_sha256")
            for e in AuditLog(self.ws.audit_log).entries()
            if e.event == "serve.deployed"
            and e.subjects.get("model") == f"{name}:{number}"
            and e.payload.get("backend") == self.backend.name
        ]
        if not logged or logged[-1] != sha256_file(path):
            raise IntegrityError(f"deployment record of {name}:{number} does not match the log")
        if record["manifest_sha256"] != sha256_file(
            self.registry.path(number, name) / "manifest.json"
        ):
            raise IntegrityError(f"{name}:{number} was deployed from another manifest")
        deploy.check_export(Path(record["export_dir"]))
        if record["backend_digest"] is not None:
            served = self.backend.digest(record["backend_model"])
            if served != record["backend_digest"]:
                raise IntegrityError(
                    f"{self.backend.name} serves {record['backend_model']} with digest "
                    f"{served}, deployed {record['backend_digest']}"
                )
        same = previous is not None and previous["version"] == number
        if not same:
            # A new version (first start, promotion or rollback): new metrics label,
            # fresh drift window, and an audit entry saying what is now live.
            self.metrics.live.clear()
            self.metrics.live.labels(name, str(number), self.backend.name).set(1)
            self.monitor = drift.DriftMonitor(
                record["drift_reference"],
                self.task,
                drift.thresholds_from(self.ws.section("monitoring")),
            )
            AuditLog(self.ws.audit_log).append(
                "serve.live",
                subjects={"model": f"{name}:{number}"},
                payload={
                    "backend": self.backend.name,
                    "backend_digest": record["backend_digest"],
                    "replaced": f"{name}:{previous['version']}" if previous else None,
                },
            )
            log.info("serving %s:%s via %s (verified)", name, number, self.backend.name)
        return record


class Gateway:
    """Request handling, metrics and drift alerts around a :class:`Live` model."""

    def __init__(self, ws: Workspace, backend: backends.Backend) -> None:
        self.ws = ws
        self.metrics = Metrics()
        self.live = Live(ws, backend, self.metrics)
        self.tracer = _tracer()
        self.last_alert: dict[str, float] = {}
        self.proposal: tuple[float, Path] | None = None
        self.cooldown = float(ws.section("monitoring").get("alert_cooldown_seconds", 3600))

    def triage(self, text: str) -> tuple[int, dict[str, Any]]:
        """Answer one ticket. Returns (HTTP status, body)."""
        self.live.refresh()
        current = self.live.current
        if current is None:
            self.metrics.requests.labels("refused").inc()
            return 503, {"error": "no verified model to serve", "detail": self.live.error}
        started = time.monotonic()
        span = self.tracer.start_as_current_span("triage") if self.tracer else None
        try:
            if span:
                span.__enter__()
            answer = self.live.backend.complete(current["backend_model"], text, 24)
        except LineageError as exc:
            self.metrics.requests.labels("backend_error").inc()
            return 502, {"error": str(exc)}
        finally:
            if span:
                span.__exit__(None, None, None)
        output = answer.text.split("###")[0].strip()
        self.metrics.latency.observe(time.monotonic() - started)
        self.metrics.tokens.labels("prompt").inc(answer.prompt_tokens)
        self.metrics.tokens.labels("completion").inc(answer.completion_tokens)
        task = self.live.task
        problems = task.problems(output)
        if problems:
            self.metrics.invalid.inc()
        self.metrics.requests.labels("ok").inc()
        if self.live.monitor is not None:
            self.live.monitor.observe(text, output)
            self._check_drift(current)
        body: dict[str, Any] = {
            **task.parse(output),
            "valid": not problems,
            "model": f"{current['model']}:{current['version']}",
        }
        return 200, body

    def _check_drift(self, current: dict[str, Any]) -> None:
        assert self.live.monitor is not None
        result = self.live.monitor.scores()
        for signal, value in result["scores"].items():
            self.metrics.drift.labels(signal).set(value)
        now = time.monotonic()
        fresh = [
            b
            for b in result["breaches"]
            if now - self.last_alert.get(b["signal"], -self.cooldown) >= self.cooldown
        ]
        if not fresh:
            return
        for breach in fresh:
            self.last_alert[breach["signal"]] = now
            self.metrics.alerts.labels(breach["signal"]).inc()
        # One proposal per model per cooldown window; later breaches in the same window
        # are logged against it rather than producing a pile of near-identical files.
        if self.proposal and now - self.proposal[0] < self.cooldown:
            AuditLog(self.ws.audit_log).append(
                "monitor.drift_alert",
                subjects={"model": f"{current['model']}:{current['version']}"},
                payload={"breaches": fresh, "proposal": self.proposal[1].name},
            )
            return
        self.proposal = (now, propose_retraining(self.ws, current, result, fresh))


def propose_retraining(
    ws: Workspace, current: dict[str, Any], result: dict[str, Any], breaches: list[dict[str, Any]]
) -> Path:
    """Write a retraining proposal and log the alert. Never starts training."""
    ws.proposals.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    path = ws.proposals / f"retrain-{current['model']}-v{current['version']}-{stamp}.json"
    proposal = {
        "status": "proposed",
        "model": f"{current['model']}:{current['version']}",
        "raised_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "window_samples": result["samples"],
        "breaches": breaches,
        "scores": result["scores"],
        "suggested_steps": [
            "sample and label recent tickets (the gateway keeps no request text)",
            "lineage data ingest --name <name> train=<new train> heldout=<new heldout>",
            "lineage data validate <version>; lineage data plant-canaries <version>",
            "lineage train run <version>; lineage eval run <run>",
            "lineage registry register <run>; promote through staging with an approval",
        ],
        "note": "Lineage never retrains on its own. A person decides whether this drift "
        "is a real change in the tickets or a problem upstream.",
    }
    path.write_text(json.dumps(proposal, indent=2) + "\n")
    log.warning(
        "drift alert on %s: %s", proposal["model"], ", ".join(b["signal"] for b in breaches)
    )
    AuditLog(ws.audit_log).append(
        "monitor.drift_alert",
        subjects={"model": proposal["model"]},
        payload={"breaches": breaches, "proposal": path.name, "proposal_sha256": sha256_file(path)},
    )
    return path


def handler(gateway: Gateway) -> type[BaseHTTPRequestHandler]:
    """Bind a request handler class to a gateway."""

    class Handler(BaseHTTPRequestHandler):
        server_version = "lineage-gateway"

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, data: dict[str, Any]) -> None:
            self._send(status, json.dumps(data).encode(), "application/json")

        def do_GET(self) -> None:
            if self.path == "/metrics":
                self._send(200, gateway.metrics.render(), "text/plain; version=0.0.4")
            elif self.path == "/healthz":
                gateway.live.refresh()
                current = gateway.live.current
                if current is None:
                    self._json(503, {"status": "refusing", "detail": gateway.live.error})
                else:
                    self._json(
                        200,
                        {
                            "status": "serving",
                            "model": f"{current['model']}:{current['version']}",
                            "backend": current["backend"],
                        },
                    )
            elif self.path == "/drift":
                monitor = gateway.live.monitor
                self._json(200, monitor.scores() if monitor else {"samples": 0})
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self) -> None:
            if self.path != "/v1/triage":
                self._json(404, {"error": "not found"})
                return
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0 or length > MAX_BODY:
                self._json(413 if length > MAX_BODY else 400, {"error": "bad body size"})
                return
            try:
                payload = json.loads(self.rfile.read(length))
                text = str(payload["ticket"])
            except (ValueError, KeyError, TypeError):
                self._json(400, {"error": 'expected JSON {"ticket": "..."}'})
                return
            status, body = gateway.triage(text)
            self._json(status, body)

        def log_message(self, format: str, *args: Any) -> None:
            log.debug("%s %s", self.address_string(), format % args)

    return Handler


def serve(ws: Workspace, host: str, port: int, backend_name: str | None = None) -> None:
    """Run the gateway until interrupted."""
    settings = ws.section("serve")
    backend = backends.make(
        backend_name or str(settings.get("backend", "ollama")), settings.get("backend_url")
    )
    gateway = Gateway(ws, backend)
    gateway.live.refresh()
    if gateway.live.current is None:
        raise PolicyDenied(f"refusing to start: {gateway.live.error}")
    server = ThreadingHTTPServer((host, port), handler(gateway))
    log.info("listening on http://%s:%s", host, port)
    try:
        server.serve_forever()
    finally:
        server.server_close()
