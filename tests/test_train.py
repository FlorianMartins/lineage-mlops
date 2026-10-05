import json

import pytest

from lineage.audit import AuditLog
from lineage.data import service as data_service
from lineage.errors import IntegrityError, PolicyDenied
from lineage.train import service
from lineage.train.config import TrainConfig

pytestmark = pytest.mark.ml


def _ready(ws, version):
    data_service.run_validation(ws, version)
    return version


def test_config_rejects_unknown_keys_and_bad_values():
    with pytest.raises(Exception, match="unknown keys"):
        TrainConfig.from_config({"epocs": 3})
    with pytest.raises(Exception, match="loss_on"):
        TrainConfig.from_config({}, loss_on="everything")
    assert TrainConfig.from_config({"epochs": 3}, epochs=None).epochs == 3
    assert TrainConfig.from_config({"epochs": 3}, epochs=5).epochs == 5


def test_training_refuses_unvalidated_data(tiny_workspace, clean_version):
    with pytest.raises(PolicyDenied, match="never been validated"):
        service.train(tiny_workspace, clean_version)


def test_training_refuses_a_tampered_base_model(tiny_workspace, clean_version):
    _ready(tiny_workspace, clean_version)
    snapshot = next(tiny_workspace.models.iterdir())
    with (snapshot / "tokenizer.json").open("a") as handle:
        handle.write(" ")
    with pytest.raises(IntegrityError, match=r"tokenizer\.json"):
        service.train(tiny_workspace, clean_version)


def test_run_is_recorded_everywhere_and_reproducible(tiny_workspace, clean_version):
    _ready(tiny_workspace, clean_version)
    first = service.train(tiny_workspace, clean_version)
    second = service.train(tiny_workspace, clean_version)

    # Same config hash, same bytes.
    assert first.record["config_hash"] == second.record["config_hash"]
    assert first.record["adapter_files"] == second.record["adapter_files"]
    assert first.id != second.id
    assert set(first.record["adapter_files"]) == {
        "adapter_config.json",
        "adapter_model.safetensors",
    }
    adapter_config = json.loads((first.adapter / "adapter_config.json").read_text())
    assert adapter_config["base_model_name_or_path"].startswith("test/tiny-llama@")
    assert first.record["metrics"]["final_loss"] < first.record["metrics"]["first_loss"]

    # MLflow has the run with the lineage tags.
    import mlflow

    mlflow.set_tracking_uri(first.record["mlflow"]["tracking_uri"])
    tracked = mlflow.get_run(first.record["mlflow"]["run_id"])
    assert tracked.data.tags["lineage.config_hash"] == first.record["config_hash"]
    assert tracked.data.tags["lineage.dataset"] == clean_version
    assert "train_loss" in tracked.data.metrics

    # The audit log has start and finish, linked to the dataset and base model.
    finished = [
        e for e in AuditLog(tiny_workspace.audit_log).entries() if e.event == "train.finished"
    ]
    assert finished[0].subjects == {
        "run": first.id,
        "dataset": clean_version,
        "base_model": first.record["base_model"],
    }
    assert finished[0].payload["adapter_files"] == first.record["adapter_files"]

    # A changed hyperparameter is a different experiment.
    third = service.train(tiny_workspace, clean_version, seed=99)
    assert third.record["config_hash"] != first.record["config_hash"]
    assert third.record["adapter_files"] != first.record["adapter_files"]

    # Loading a run re-verifies the adapter.
    assert service.load_run(tiny_workspace, first.id).id == first.id
    with (first.adapter / "adapter_model.safetensors").open("r+b") as handle:
        handle.seek(-1, 2)
        handle.write(b"\x07")
    with pytest.raises(IntegrityError):
        service.load_run(tiny_workspace, first.id)


def test_completion_only_loss_masks_the_prompt(tiny_workspace):
    from lineage.train import lora

    _, tokenizer = lora.load_base(next(tiny_workspace.models.iterdir()))
    task = data_service.task_of(tiny_workspace)
    ids, labels = lora.encode(
        tokenizer, task, "VPN is down", "category: network\npriority: high", TrainConfig()
    )
    masked = labels.count(-100)
    assert 0 < masked < len(ids) and labels[-1] == tokenizer.eos_token_id
    _, full = lora.encode(
        tokenizer,
        task,
        "VPN is down",
        "category: network\npriority: high",
        TrainConfig(loss_on="full"),
    )
    assert -100 not in full
