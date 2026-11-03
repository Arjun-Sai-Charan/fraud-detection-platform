import json

import numpy as np
import pandas as pd

from src.monitoring.drift import (
    _evidently_summary,
    _run_evidently,
    build_shifted_test,
    make_report,
    natural_baseline_from_windows,
    natural_trigger_threshold,
)
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
        def __init__(self, version, tags=None):
            self.version = str(version)
            self.tags = tags or {}

    class Client:
        versions = [Version(1), Version(2), Version(3, {"retraining_role": "candidate"}), Version(4, {"retraining_role": "candidate"})]

        def get_model_version_by_alias(self, _name, alias):
            if alias == "candidate":
                return Version(4)
            raise MlflowException("alias not found")

        def search_model_versions(self, _query):
            return self.versions

        def get_model_version(self, _name, version):
            return Version(version)

    client = Client()
    assert serving_model_version(client, "fraud-xgb").version == "2"
    assert serving_model_version(client, "fraud-xgb", "4").version == "4"


def test_scorer_filters_only_the_feast_relisting_warning(caplog):
    import logging

    from src.serving.scorer import FraudScorer, _FEAST_DEPRECATION

    FraudScorer(model=object(), feature_store=object(), feature_list=[], threshold={})
    with caplog.at_level(logging.WARNING):
        logging.warning(_FEAST_DEPRECATION)
        logging.warning("unrelated warning must remain visible")
    messages = [record.getMessage() for record in caplog.records]
    assert _FEAST_DEPRECATION not in messages
    assert "unrelated warning must remain visible" in messages


def test_promote_candidate_dry_run_does_not_set_champion_alias():
    from scripts.promote_candidate import promote_candidate
    from mlflow.exceptions import MlflowException

    class Version:
        version = "4"
        tags = {"retraining_role": "candidate"}

    class Client:
        def __init__(self):
            self.alias_updates = []

        def get_model_version(self, _name, _version):
            return Version()

        def get_model_version_by_alias(self, _name, alias):
            if alias == "candidate":
                return Version()
            raise MlflowException("alias missing")

        def set_registered_model_alias(self, *args):
            self.alias_updates.append(args)

    client = Client()
    result = promote_candidate("4", dry_run=True, client=client)
    assert result["dry_run"] is True
    assert result["promoted"] is False
    assert client.alias_updates == []


def test_natural_baseline_windows_threshold_and_shifted_stress():
    rng = np.random.default_rng(12)
    names = ["TransactionAmt", "txn_count_last_1h", "txn_count_last_24h", "device_change_vs_previous", "D1"]
    windows = []
    for offset in (0.0, 0.1, 0.2, 0.3):
        windows.append(pd.DataFrame({
            "TransactionAmt": rng.normal(10 + offset, 1, 300),
            "txn_count_last_1h": rng.poisson(1 + offset, 300).astype(float),
            "txn_count_last_24h": rng.poisson(3 + offset, 300).astype(float),
            "device_change_vs_previous": rng.binomial(1, 0.1 + offset / 10, 300).astype(float),
            "D1": rng.normal(5 + offset, 2, 300),
        }))
    baseline = natural_baseline_from_windows(windows, names)
    assert baseline["comparison_count"] == 3
    assert np.isclose(baseline["mean"], np.mean(baseline["drift_shares"]))
    assert np.isclose(baseline["max"], np.max(baseline["drift_shares"]))
    assert np.isclose(baseline["p95"], np.percentile(baseline["drift_shares"], 95))
    threshold = natural_trigger_threshold(baseline["max"], 0.1)
    assert threshold > baseline["max"]

    shifted = build_shifted_test(windows[-1], names, stress_multiplier=100.0)
    summary = _evidently_summary(_run_evidently(windows[-2], shifted, names), 300, 300)
    assert summary["drift_share"] > threshold
    assert any("cumulative_by_construction" in row["tags"] for row in summary["drifted_column_details"])


def test_expanded_window_sizes_and_estimator_scaling():
    from src.monitoring.retrain import expanded_n_estimators, expanded_window_sizes

    sizes = expanded_window_sizes(90_000, 30_000, 30_000)
    assert sizes == {
        "original_train_rows": 90_000,
        "validation_rows_added": 30_000,
        "expanded_train_rows": 120_000,
        "test_rows": 30_000,
    }
    assert expanded_n_estimators(402, sizes["expanded_train_rows"], sizes["original_train_rows"]) == 536
