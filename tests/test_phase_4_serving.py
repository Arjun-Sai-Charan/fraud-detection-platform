import json
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from src.serving.api import create_app
from src.serving.repository import ScoreRepository
from src.serving.scorer import FraudScorer
from src.serving.consumer import process_transaction
from src.streaming.producer import ordered_test_rows, replay_test_split


class FakeOnlineResponse:
    def to_dict(self):
        return {
            "transaction_features__transaction_dt": [123456],
            "transaction_features__TransactionAmt": [25.0],
        }


class FakeFeatureStore:
    def __init__(self):
        self.last_request = None

    def get_online_features(self, **kwargs):
        self.last_request = kwargs
        return FakeOnlineResponse()


class FakeModel:
    def __init__(self):
        self.last_frame = None

    def predict_proba(self, frame):
        self.last_frame = frame.copy()
        return np.array([[0.15, 0.85]])


def test_scorer_reads_feast_online_features_and_returns_phase5_schema():
    feature_store = FakeFeatureStore()
    model = FakeModel()
    scorer = FraudScorer(
        model=model,
        feature_store=feature_store,
        feature_list=["transaction_dt", "TransactionAmt"],
        threshold={"threshold": 0.5, "tie_break_transaction_id_max": 10},
    )
    result = scorer.score(42)
    assert feature_store.last_request["entity_rows"] == [{"transaction_id": 42}]
    assert feature_store.last_request["features"] == [
        "transaction_features:transaction_dt",
        "transaction_features:TransactionAmt",
    ]
    assert model.last_frame.iloc[0].tolist() == [123456.0, 25.0]
    assert result["xgb_score"] == 0.85
    assert result["alert"] is True
    assert result["final_score"] == result["xgb_score"]
    assert result["gnn_score"] is None
    assert result["ring_id"] is None and result["ring_size"] is None
    assert result["latency_ms"] >= 0


def test_api_score_alert_feedback_and_metrics_round_trip(tmp_path):
    class MockScorer:
        def score(self, transaction_id):
            return {
                "transaction_id": transaction_id,
                "xgb_score": 0.91,
                "alert": True,
                "gnn_score": None,
                "final_score": 0.91,
                "ring_id": None,
                "ring_size": None,
                "latency_ms": 4.25,
            }

    client = TestClient(create_app(MockScorer(), ScoreRepository(tmp_path / "api.sqlite")))
    assert client.get("/health").json() == {"status": "ok"}
    scored = client.post("/score", json={"transaction_id": 5001})
    assert scored.status_code == 200
    assert scored.json()["alert"] is True
    assert scored.json()["gnn_score"] is None
    assert len(client.get("/alerts?status=open").json()["alerts"]) == 1

    feedback = client.post(
        "/feedback",
        json={"transaction_id": 5001, "analyst_label": "fraud", "case_status": "closed"},
    )
    assert feedback.status_code == 200
    assert feedback.json()["analyst_label"] == "fraud"
    assert client.get("/alerts?status=open").json()["alerts"] == []
    assert len(client.get("/alerts?status=closed").json()["alerts"]) == 1
    metrics = client.get("/metrics-lite").json()
    assert metrics["score_count"] == 1
    assert metrics["alert_count"] == 1
    assert metrics["feedback_count"] == 1
    assert metrics["latency_ms"]["p50"] == 4.25


class FakeProducer:
    def __init__(self):
        self.messages = []

    def produce(self, topic, *, key, value):
        self.messages.append((topic, key, json.loads(value)))

    def poll(self, _timeout):
        return 0

    def flush(self, _timeout=0):
        return 0


def test_producer_orders_test_rows_before_limit_and_replays_payloads():
    frame = pd.DataFrame(
        {
            "transaction_id": [4, 1, 3, 2],
            "event_timestamp": pd.to_datetime(
                ["2018-01-02T00:00:00Z", "2018-01-01T00:00:00Z", "2018-01-01T00:00:00Z", "2018-01-03T00:00:00Z"]
            ),
            "split": ["test", "test", "test", "train"],
        }
    )
    assert ordered_test_rows(frame, limit=2).transaction_id.tolist() == [1, 3]
    producer = FakeProducer()
    sent = replay_test_split(frame, producer, "transactions", limit=3)
    assert sent == 3
    assert [message[2]["transaction_id"] for message in producer.messages] == [1, 3, 4]


def test_consumer_scores_persists_and_publishes_with_mock_kafka(tmp_path):
    class MockScorer:
        def score(self, transaction_id):
            return {
                "transaction_id": transaction_id,
                "xgb_score": 0.8,
                "alert": True,
                "gnn_score": None,
                "final_score": 0.8,
                "ring_id": None,
                "ring_size": None,
                "latency_ms": 2.0,
            }

    producer = FakeProducer()
    repository = ScoreRepository(tmp_path / "consumer.sqlite")
    result = process_transaction({"transaction_id": 71}, MockScorer(), repository, producer)
    assert result["alert"] is True
    assert producer.messages[0][0] == "fraud-scores"
    assert producer.messages[0][2]["transaction_id"] == 71
    assert repository.metrics_lite()["score_count"] == 1


def test_export_registered_model_copies_local_assets(tmp_path, monkeypatch):
    from scripts import export_model

    class Version:
        name = "fraud-xgb"
        version = "7"
        run_id = "run-7"
        source = "file:///model"

    class FakeClient:
        def __init__(self, tracking_uri):
            assert tracking_uri == "file:./mock"

        def search_model_versions(self, _filter):
            return [Version()]

        def download_artifacts(self, run_id, path, dst_path=None):
            assert run_id == "run-7" and path == "model"
            model_dir = Path(dst_path) / "model"
            model_dir.mkdir(parents=True)
            (model_dir / "MLmodel").write_text("mock")
            (model_dir / "serving_threshold.json").write_text(json.dumps({"threshold": 0.5}))
            (model_dir / "feature_list.json").write_text(json.dumps({"features": ["amount"]}))
            return str(model_dir)

    monkeypatch.setattr(export_model, "MlflowClient", FakeClient)
    out = tmp_path / "artifacts" / "xgb"
    result = export_model.export_registered_model("file:./mock", output_dir=out)
    assert result["version"] == 7
    assert (out / "model" / "MLmodel").is_file()
    assert json.loads((out / "threshold.json").read_text())["threshold"] == 0.5
    assert json.loads((out / "feature_list.json").read_text())["features"] == ["amount"]
