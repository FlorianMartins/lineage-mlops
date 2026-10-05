import json
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from lineage.audit import AuditLog
from lineage.data import service as data_service
from lineage.errors import LineageError, PolicyDenied
from lineage.serve import backends, drift
from lineage.serve.deploy import legacy_rope_keys
from lineage.task import TaskSpec
from tests.test_registry import promotable  # noqa: F401 - shared fixture

TASK = TaskSpec("t", {"category": ("network", "billing"), "priority": ("low", "high")})


# -- drift ------------------------------------------------------------------
def test_psi_and_js_basics():
    assert drift.psi([0.5, 0.5], [0.5, 0.5]) == 0
    assert drift.psi([0.5, 0.5], [0.9, 0.1]) > 0.25
    assert drift.js({"a": 1.0}, {"a": 1.0}) == 0
    assert drift.js({"a": 1.0}, {"b": 1.0}) == pytest.approx(1.0)


def _reference():
    inputs = [f"my vpn drops every {i} minutes on the network" for i in range(60)] + [
        f"invoice {i} charged twice please refund" for i in range(60)
    ]
    outputs = ["category: network\npriority: high"] * 60 + ["category: billing\npriority: low"] * 60
    return inputs, outputs, drift.build_reference(inputs, outputs, TASK)


def test_no_drift_on_reference_like_traffic():
    inputs, outputs, reference = _reference()
    monitor = drift.DriftMonitor(reference, TASK, drift.Thresholds(window=100, min_samples=20))
    for text, out in zip(inputs[::2], outputs[::2], strict=True):
        monitor.observe(text, out)
    result = monitor.scores()
    assert result["ready"] and result["breaches"] == []


def test_drift_on_shifted_traffic_and_not_before_min_samples():
    _, _, reference = _reference()
    monitor = drift.DriftMonitor(reference, TASK, drift.Thresholds(window=50, min_samples=30))
    for i in range(29):
        monitor.observe(f"Bonjour, le portail RH {i} n'affiche plus mes congés depuis lundi", "?")
    assert not monitor.scores()["ready"] and monitor.scores()["breaches"] == []
    monitor.observe("Bonjour, encore le portail RH", "nonsense")
    signals = {b["signal"] for b in monitor.scores()["breaches"]}
    assert {"input_vocabulary_js", "output_category_js"} <= signals


def test_monitor_never_keeps_request_text():
    _, _, reference = _reference()
    monitor = drift.DriftMonitor(reference, TASK, drift.Thresholds())
    monitor.observe("secret ticket from alice@example.com", "category: network\npriority: low")
    assert "alice" not in repr(list(monitor.window))


def test_thresholds_from_config():
    t = drift.thresholds_from({"window": "50", "length_psi": 0.1, "unrelated": 1})
    assert t.window == 50 and t.length_psi == 0.1


def test_legacy_rope_keys(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"rope_parameters": {"rope_theta": 100000, "rope_type": "default"}}))
    legacy_rope_keys(path)
    assert json.loads(path.read_text())["rope_theta"] == 100000


# -- backends against a fake HTTP server ---------------------------------------
class _Fake(BaseHTTPRequestHandler):
    def do_GET(self):
        body = json.dumps({"models": [{"name": "m:v1", "digest": "abc"}]}).encode()
        self.send_response(200)
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/api/generate":
            body = {"response": "category: network", "prompt_eval_count": 7, "eval_count": 3}
        else:
            body = {
                "choices": [{"text": f"echo {request['prompt']}"}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2},
            }
        self.send_response(200)
        self.end_headers()
        self.wfile.write(json.dumps(body).encode())

    def log_message(self, *args):
        pass


@pytest.fixture
def fake_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Fake)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


def test_ollama_backend(fake_server):
    ollama = backends.Ollama(fake_server)
    assert ollama.digest("m:v1") == "abc" and ollama.digest("other") is None
    out = ollama.complete("m:v1", "x", 8)
    assert (out.text, out.prompt_tokens, out.completion_tokens) == ("category: network", 7, 3)


def test_vllm_backend(fake_server, tmp_path):
    vllm = backends.VLLM(fake_server)
    assert vllm.complete("m", "hi", 4).text == "echo hi"
    deployed = vllm.deploy(tmp_path, "m")
    assert deployed["digest"] is None and "vllm serve" in (tmp_path / "vllm-serve.sh").read_text()


def test_unreachable_backend_and_unknown_backend():
    with pytest.raises(LineageError, match="unreachable"):
        backends.Ollama("http://127.0.0.1:9").digest("x")
    with pytest.raises(LineageError):
        backends.make("triton", None)


# -- deployment and gateway on the tiny model ------------------------------------
class TorchBackend:
    """Serves export directories with transformers, like a backend would."""

    name = "fake"

    def __init__(self, task):
        self.task = task
        self.models = {}
        self.digests = {}
        self.oracle = None  # prompt -> answer: a served model unlike the evaluated one

    def deploy(self, export_dir, model):
        from lineage.evaluate.predictor import Model

        self.models[model] = Model.load(export_dir, None, model, threads=1)
        self.digests[model] = "digest-" + model
        return {"model": model, "digest": self.digests[model]}

    def digest(self, model):
        return self.digests.get(model)

    def complete(self, model, prompt, max_tokens):
        if self.oracle is not None:
            return backends.Completion(self.oracle.get(prompt, "?"), 3, 2)
        text = self.models[model].generate([prompt], max_new_tokens=max_tokens, batch=1)[0]
        return backends.Completion(text, 3, 2)


@pytest.fixture
def served(promotable, monkeypatch):  # noqa: F811 - the fixture imported above
    """Two versions registered, v1 in production, both deployed on the fake backend."""
    from lineage.registry import service as registry_service
    from lineage.serve import deploy

    ws, runs = promotable
    backend = TorchBackend(data_service.task_of(ws))
    monkeypatch.setattr(backends, "make", lambda *_: backend)
    for run in runs:
        number = registry_service.register(ws, run)
        registry_service.promote(ws, f"ticket-triage:{number}", "staging")
        monkeypatch.setenv("LINEAGE_ACTOR", "bob")
        registry_service.approve(ws, f"ticket-triage:{number}", "reviewed the evaluation")
        monkeypatch.setenv("LINEAGE_ACTOR", "alice")
        deploy.deploy(ws, f"ticket-triage:{number}")
    registry_service.promote(ws, "ticket-triage:1", "production")
    return ws, backend


@pytest.mark.ml
@pytest.mark.tools
def test_deploy_records_parity_and_digest(served):
    ws, _ = served
    record = json.loads((ws.state / "deployments" / "ticket-triage-v1-fake.json").read_text())
    assert (
        record["parity"]["passed"] and record["backend_digest"] == "digest-lineage-ticket-triage:v1"
    )
    assert record["parity"]["served_exact_match"] == record["parity"]["evaluated_exact_match"]
    assert json.loads((Path(record["export_dir"]) / "config.json").read_text()).get("rope_theta")
    events = [e.event for e in AuditLog(ws.audit_log).entries()]
    assert events.count("serve.deployed") == 2 and events.count("serve.exported") == 2


@pytest.mark.ml
@pytest.mark.tools
def test_deploy_refuses_when_the_served_model_does_not_match(served, monkeypatch):
    from lineage.serve import deploy

    ws, backend = served
    # The random tiny model scores 0 when evaluated; a served model that answers every
    # held-out ticket correctly is just as different (parity checks the gap both ways).
    task = data_service.task_of(ws)
    heldout = (ws.root / "data" / "heldout.jsonl").read_text().splitlines()
    backend.oracle = {
        task.prompt(json.loads(line)["input"]): json.loads(line)["output"] for line in heldout
    }
    with pytest.raises(PolicyDenied, match="conversion changed the model"):
        deploy.deploy(ws, "ticket-triage:1")
    assert any(e.event == "serve.deploy_denied" for e in AuditLog(ws.audit_log).entries())


@pytest.mark.ml
@pytest.mark.tools
def test_gateway_serves_follows_rollback_and_refuses_tampering(served):
    from lineage.registry import service as registry_service
    from lineage.serve.gateway import Gateway

    ws, backend = served
    gateway = Gateway(ws, backend)
    status, body = gateway.triage("My VPN drops every few minutes.")
    assert status == 200 and body["model"] == "ticket-triage:1" and "valid" in body
    metrics = gateway.metrics.render().decode()
    assert 'lineage_requests_total{status="ok"} 1.0' in metrics
    assert 'lineage_model_info{backend="fake",model="ticket-triage",version="1"} 1.0' in metrics

    # Promotion of v2 then rollback: the gateway follows the registry, re-verifying.
    registry_service.promote(ws, "ticket-triage:2", "production")
    assert gateway.triage("VPN down")[1]["model"] == "ticket-triage:2"
    registry_service.rollback(ws, None, "v2 misroutes security tickets")
    assert gateway.triage("VPN down")[1]["model"] == "ticket-triage:1"

    # Someone re-tags the backend model: refused at the next verification.
    backend.digests["lineage-ticket-triage:v1"] = "something-else"
    gateway.live.verified_at = 0
    status, body = gateway.triage("VPN down")
    assert status == 503 and "digest" in body["detail"]
    events = [e.event for e in AuditLog(ws.audit_log).entries()]
    assert "serve.refused" in events and events.count("serve.live") == 3


@pytest.mark.ml
@pytest.mark.tools
def test_gateway_refuses_a_tampered_export(served):
    from lineage.serve.gateway import Gateway

    ws, backend = served
    record = json.loads((ws.state / "deployments" / "ticket-triage-v1-fake.json").read_text())
    with (Path(record["export_dir"]) / "config.json").open("a") as handle:
        handle.write(" ")
    status, body = Gateway(ws, backend).triage("VPN down")
    assert status == 503 and "changed after it was built" in body["detail"]


@pytest.mark.ml
@pytest.mark.tools
def test_drift_alert_writes_one_proposal_and_never_trains(served):
    from lineage.serve.gateway import Gateway

    ws, backend = served
    text = (ws.root / "lineage.toml").read_text()
    (ws.root / "lineage.toml").write_text(
        text.replace("min_samples = 50", "min_samples = 10").replace("window = 200", "window = 20")
    )
    from lineage.workspace import Workspace

    ws = Workspace.load(ws.root)
    gateway = Gateway(ws, backend)
    runs_before = sorted(p.name for p in ws.runs.iterdir())
    for i in range(25):
        gateway.triage(f"Bonjour, le portail RH {i} n'affiche plus mes congés depuis lundi matin")
    proposals = list(ws.proposals.glob("retrain-*.json"))
    assert len(proposals) == 1
    proposal = json.loads(proposals[0].read_text())
    assert proposal["status"] == "proposed" and proposal["breaches"]
    assert sorted(p.name for p in ws.runs.iterdir()) == runs_before  # nothing trained
    assert "lineage_drift_alerts_total" in gateway.metrics.render().decode()


@pytest.mark.ml
@pytest.mark.tools
def test_http_endpoints(served):
    from lineage.serve.gateway import Gateway, handler

    ws, backend = served
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler(Gateway(ws, backend)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        health = json.loads(urllib.request.urlopen(f"{base}/healthz").read())
        assert health == {"status": "serving", "model": "ticket-triage:1", "backend": "fake"}
        request = urllib.request.Request(
            f"{base}/v1/triage", data=json.dumps({"ticket": "VPN down"}).encode()
        )
        assert "model" in json.loads(urllib.request.urlopen(request).read())
        for data, code in ((b"not json", 400), (b"x" * 20000, 413)):
            with pytest.raises(urllib.error.HTTPError) as error:
                urllib.request.urlopen(urllib.request.Request(f"{base}/v1/triage", data=data))
            assert error.value.code == code
        assert b"lineage_request_seconds" in urllib.request.urlopen(f"{base}/metrics").read()
    finally:
        server.shutdown()
