"""SageMaker entry point: rebuild the exact dataset version, verify the base model, train.

Content addressing makes the hand-over checkable: the job re-ingests the uploaded
splits and must arrive at the *same* dataset version id the local plan named, and the
base model channel must match the pin file by file. Anything else stops the job before
training. The run directory (adapter + run.json) is written to /opt/ml/model, which
SageMaker uploads (KMS-encrypted) to the plan's output prefix.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

from lineage.data import service as data_service
from lineage.supply.models import LocalHub, ModelPin, fetch
from lineage.train import service as train_service
from lineage.workspace import Workspace

INPUT = Path("/opt/ml/input")
OUTPUT = Path("/opt/ml/model")


def main() -> int:
    """Rebuild, verify, train; non-zero exit stops the job."""
    params = json.loads((INPUT / "config" / "hyperparameters.json").read_text())
    expected = params["lineage_dataset"]
    ws = Workspace.load(Path("/opt/lineage/examples/triage"))
    data = INPUT / "data" / "data"
    splits = {p.stem: p for p in sorted(data.glob("*.jsonl"))}
    version, _ = data_service.ingest(ws, "cloud", splits)
    if version != expected:
        print(f"dataset hashes to {version}, plan named {expected}: refusing", file=sys.stderr)
        return 1
    data_service.run_validation(ws, version)
    pin = ModelPin.from_config(ws.section("base_model"))
    fetch(ws, LocalHub(INPUT / "data" / "model", pin.licence))
    run = train_service.train(ws, version)
    shutil.copytree(run.path, OUTPUT / run.id)
    shutil.copy2(ws.audit_log, OUTPUT / run.id / "cloud-audit.jsonl")
    print(f"trained {run.id} on {version}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
