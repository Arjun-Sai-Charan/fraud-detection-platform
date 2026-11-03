import json

import numpy as np
import pandas as pd

from src.monitoring.drift import build_shifted_test, make_report
from src.monitoring.retrain import run_retraining, should_trigger
from src.models.train_xgb import _prepare_training_splits
from src.common.model_registry import serving_model_version
from mlflow.exceptions import MlflowException
from src.serving.repository import ScoreRepository


def test_evidently_drift_share_and_deliberately_shifted_slice(tmp_path):
    rng = np.random.default_rng(42)
    reference = pd.DataFrame({
        "TransactionAmt": rng.normal(20, 3, 250),
        "txn_count_last_1h": rng.poisson(1, 250).astype(float),
        "txn_count_last_24h": rng.poisson(3, 250).astype(float),
        "device_change_vs_previous": rng.binomial(1, 0.05, 250).astype(float),
    })
    current = reference.copy()
    shifted = build_shifted_test(current, list(current.columns))
    shifted_summary = make_report(
        reference,
        shifted,
        columns=list(reference.columns),
        report_name="shifted",
        output_dir=tmp_path,
        drift_share_threshold=0.3,
    )
    assert shifted_summary["drift_share"] > 0.3
    assert shifted_summary["dataset_drift"] is True
    assert should_trigger(shifted_summary["drift_share"], 0, drift_threshold=0.3)[0] is True
    assert "TransactionAmt" in shifted_summary["drifted_columns"]
    assert shifted_summary["test_names"]
    assert (tmp_path / "shifted.json").exists()
    assert (tmp_path / "shifted.html").exists()


def test_trigger_stays_off_below_threshold():
    assert should_trigger(0.3, 0, drift_threshold=0.3, min_feedback=25) == (False, [])


def test_dry_run_does_not_call_training_or_change_model(tmp_path):
    report = tmp_path / "drift.json"
    report.write_text(json.dumps({"drift_share": 0.8}), encoding="utf-8")
    calls = []

    def forbidden_train(*_args, **_kwargs):
        calls.append(True)
        raise AssertionError("dry run must not train")

    result = run_retraining(
        dry_run=True,
        tracking_uri="file:./should_not_be_created",
        drift_report=report,
        feedback_db=tmp_path / "absent.db",
        output_path=tmp_path / "retrain_log.json",
        train_callable=forbidden_train,
    )
    assert result["triggered"] is True
    assert result["status"] == "dry_run"
    assert calls == []
    assert result["registered_model_alias"] is None
    assert not (tmp_path / "should_not_be_created").exists()


def test_feedback_count_triggers_using_temporary_sqlite(tmp_path):
    db_path = tmp_path / "serving.db"
    repository = ScoreRepository(db_path)
    for transaction_id in range(1, 26):
        repository.save_score({
            "transaction_id": transaction_id,
            "xgb_score": 0.9,
            "alert": True,
            "gnn_score": None,
            "final_score": 0.9,
            "ring_id": None,
            "ring_size": None,
            "latency_ms": 1.0,
        })
        repository.add_feedback(transaction_id, "fraud", "closed")
    result = run_retraining(
        dry_run=True,
        drift_report=tmp_path / "missing-drift.json",
        feedback_db=db_path,
        output_path=tmp_path / "feedback-trigger.json",
        min_feedback=25,
    )
    assert result["feedback_count"] == 25
    assert result["triggered"] is True
    assert result["trigger_reasons"] == ["analyst_feedback_count=25>=25"]


def test_include_feedback_maps_confirmed_test_replay_labels(tmp_path):
    db_path = tmp_path / "serving.db"
    repository = ScoreRepository(db_path)
    repository.save_score({
        "transaction_id": 5001,
        "xgb_score": 0.9,
        "alert": True,
        "gnn_score": None,
        "final_score": 0.9,
        "ring_id": None,
        "ring_size": None,
        "latency_ms": 1.0,
    })
    repository.add_feedback(5001, "fraud", "closed")
    features_path = tmp_path / "features.parquet"
    pd.DataFrame([{
        "transaction_id": 5001,
        "split": "test",
        "isFraud": 0,
        "amount": 12.0,
    }]).to_parquet(features_path, index=False)
    from src.monitoring.retrain import load_confirmed_feedback_rows

    rows = load_confirmed_feedback_rows(db_path, features_path)
    assert len(rows) == 1
    assert int(rows.iloc[0]["isFraud"]) == 1


def test_feedback_rows_join_train_and_disable_test_evaluation():
    frame = pd.DataFrame({
        "transaction_id": [1, 2, 3, 4, 5, 6],
        "split": ["train", "train", "validation", "validation", "test", "test"],
        "isFraud": [0, 1, 0, 1, 0, 1],
        "amount": [1., 2., 3., 4., 5., 6.],
    })
    feedback = pd.DataFrame({"transaction_id": [7], "isFraud": [1], "amount": [7.]})
    splits, included = _prepare_training_splits(frame, ["amount"], feedback)
    assert included is True
    assert set(splits) == {"train", "validation"}
    assert len(splits["train"][1]) == 3
    assert int(splits["train"][1][-1]) == 1


def test_candidate_alias_is_not_selected_as_serving_model():
    class Version:
        def __init__(self, version):
            self.version = str(version)

    class Client:
        versions = [Version(1), Version(2), Version(3)]

        def get_model_version_by_alias(self, _name, alias):
            if alias == "candidate":
                return Version(3)
            raise MlflowException("alias not found")

        def search_model_versions(self, _query):
            return self.versions

        def get_model_version(self, _name, version):
            return Version(version)

    client = Client()
    assert serving_model_version(client, "fraud-xgb").version == "2"
    assert serving_model_version(client, "fraud-xgb", "3").version == "3"
