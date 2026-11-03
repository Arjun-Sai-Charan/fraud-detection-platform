from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import httpx
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from scripts.ring_precision import compare_ring_precision
from src.dashboard import data as dashboard_data
from src.serving.api import create_app
from src.serving.repository import ScoreRepository
from src.serving.scorer import FraudScorer


class DetailScorer:
    def score(self, transaction_id):
        return {
            "transaction_id": transaction_id,
            "xgb_score": 0.91,
            "alert": True,
            "gnn_score": 0.8,
            "final_score": 0.87,
            "ring_id": "ring-a",
            "ring_size": 2,
            "latency_ms": 3.0,
        }

    def feature_contributions(self, transaction_id, limit=5):
        return [{"feature": f"f{i}", "contribution": 0.5 - i / 10} for i in range(min(limit, 5))]


def test_detail_and_ring_api_endpoints_with_temp_sqlite(tmp_path):
    repository = ScoreRepository(tmp_path / "api.sqlite")
    repository.save_score(DetailScorer().score(12))
    app = create_app(
        DetailScorer(),
        repository,
        ring_data=(
            [
                {"transaction_id": 12, "ring_id": "ring-a", "ring_size": 2},
                {"transaction_id": 13, "ring_id": "ring-a", "ring_size": 2},
                {"transaction_id": 14, "ring_id": None, "ring_size": None},
            ],
            {12: 40.0, 13: 60.0},
        ),
    )
    client = TestClient(app)
    detail = client.get("/transactions/12")
    assert detail.status_code == 200
    assert detail.json()["transaction"]["xgb_score"] == 0.91
    assert len(detail.json()["top_contributions"]) == 5
    assert client.get("/transactions/404").status_code == 404
    rings = client.get("/rings").json()["rings"]
    assert rings == [{
        "ring_id": "ring-a",
        "ring_size": 2,
        "transaction_ids": [12, 13],
        "total_amount": 100.0,
        "alert_count": 1,
        "case_statuses": {"open": 1},
    }]


def test_scorer_feature_contributions_use_online_features(tmp_path):
    class Online:
        def to_dict(self):
            return {"transaction_features__f1": [1.0], "transaction_features__f2": [2.0]}

    class Store:
        last_request = None

        def get_online_features(self, **kwargs):
            self.last_request = kwargs
            return Online()

    class Booster:
        def predict(self, matrix, pred_contribs=False):
            assert pred_contribs is True
            return [[0.2, -0.8, 0.1]]

    class Model:
        def get_booster(self):
            return Booster()

    store = Store()
    scorer = FraudScorer(
        model=Model(), feature_store=store, feature_list=["f1", "f2"], threshold={"threshold": 0.5},
        gnn_artifact_dir=tmp_path,
    )
    contributions = scorer.feature_contributions(12)
    assert contributions == [
        {"feature": "f2", "contribution": -0.8},
        {"feature": "f1", "contribution": 0.2},
    ]
    assert store.last_request["entity_rows"] == [{"transaction_id": 12}]


def test_dashboard_data_shapes_empty_and_api_failure(monkeypatch):
    alerts, message = dashboard_data.shape_alerts({"alerts": []})
    assert alerts.empty and message
    rings, message = dashboard_data.shape_rings({"rings": [], "message": "No rings"})
    assert rings.empty and message == "No rings"

    def unreachable(*args, **kwargs):
        raise httpx.ConnectError("offline")

    monkeypatch.setattr(dashboard_data.httpx, "get", unreachable)
    monkeypatch.setattr(dashboard_data.httpx, "post", unreachable)
    metrics, message = dashboard_data.fetch_metrics()
    assert metrics["alert_count"] == 0 and message
    alerts, message = dashboard_data.fetch_alerts()
    assert alerts.empty and message
    transaction, message = dashboard_data.fetch_transaction(1)
    assert transaction is None and message
    feedback, message = dashboard_data.send_feedback(1, "fraud", "confirmed")
    assert feedback is None and message
    rings, message = dashboard_data.fetch_rings()
    assert rings.empty and message


def test_dashboard_data_success_and_local_status_files(monkeypatch, tmp_path):
    class Response:
        def __init__(self, value):
            self.value = value

        def raise_for_status(self):
            return None

        def json(self):
            return self.value

    monkeypatch.setattr(dashboard_data.httpx, "get", lambda url, **kwargs: Response({"alerts": [{"transaction_id": 4, "xgb_score": 0.8, "status": "open"}]}))
    alerts, message = dashboard_data.fetch_alerts()
    assert len(alerts) == 1 and message is None
    drift_path = tmp_path / "drift.json"
    drift_path.write_text(json.dumps({"drift_share": 0.1, "trigger_threshold": 0.6}))
    assert dashboard_data.latest_drift_status(drift_path)[0] == "10.0% of features; threshold 60.0%"
    retrain_path = tmp_path / "retrain.json"
    retrain_path.write_text(json.dumps({"status": "dry_run", "trigger_reasons": []}))
    assert dashboard_data.last_retrain_decision(retrain_path)[0].startswith("dry_run:")


def test_ring_precision_reports_rates_and_counts(monkeypatch, tmp_path):
    import scripts.ring_precision as ring_precision

    monkeypatch.setattr(
        ring_precision.pd,
        "read_parquet",
        lambda *args, **kwargs: pd.DataFrame({"transaction_id": [1, 2, 3], "split": ["test"] * 3, "gnn_score": [0.1, 0.1, 0.9]}),
    )
    features = pd.DataFrame({
        "transaction_id": [1, 2, 3, 4, 5],
        "split": ["test"] * 5,
        "isFraud": [1, 0, 1, 0, 0],
        "f": [1.0, 2.0, 3.0, 4.0, 5.0],
    })

    class Model:
        def predict_proba(self, frame):
            return pd.DataFrame({0: [0.1] * 5, 1: [0.9, 0.8, 0.7, 0.6, 0.5]}).to_numpy()

    rings = pd.DataFrame({"transaction_id": [1, 2, 3, 4, 5], "ring_id": ["strong", "strong", "weak", "weak", None], "ring_size": [2, 2, 2, 2, None]})
    entities = pd.DataFrame({
        "transaction_id": [1, 2, 3, 4, 5],
        "uid": ["u", "u", "u3", "u4", "u5"],
        "DeviceInfo": ["d1", "d2", "d3", "d4", "d5"],
        "P_emaildomain": ["e1", "e2", "e", "e", "e5"],
        "addr1": [10, 20, 30, 40, 50],
    })
    result = compare_ring_precision(
        features, Model(), ["f"], rings,
        {"blend_weight_xgb": 1.0, "threshold": 0.0},
        entities,
        {"uid": {"u"}, "device": set(), "email_domain": {"e"}, "addr1": set()},
    )
    assert result["flagged_transaction_count"] == 5
    breakdown = result["ring_definition_breakdown"]
    assert breakdown["ring_size_distribution"]["ring_count"] == 2
    assert breakdown["ring_size_distribution"]["median"] == 2
    assert breakdown["ring_size_distribution"]["share_of_all_flagged_rows_in_largest_ring"] == 0.4
    assert breakdown["fraud_rate_by_link_type"]["strong_links_only"]["fraud_rate"] == 0.5
    assert breakdown["fraud_rate_by_link_type"]["includes_weak_link"]["fraud_rate"] == 0.5
    assert breakdown["fraud_rate_by_link_type"]["outside_any_ring"]["fraud_rate"] == 0.0
    assert breakdown["ring_counts_by_link_type"]["only_weak_links"] == 1


def test_streamlit_apptest_smoke_when_supported():
    try:
        from streamlit.testing.v1 import AppTest
    except ImportError:
        pytest.skip("Installed Streamlit does not provide AppTest")
    app = AppTest.from_file("src/dashboard/app.py", default_timeout=15).run()
    assert not app.exception
    assert any("Fraud analyst dashboard" in title.value for title in app.title)


def test_dashboard_entrypoint_imports_helpers_from_other_working_directory(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    script_dir = repo_root / "src/dashboard"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(script_dir)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import app; from src.dashboard.data import fetch_metrics; "
            "assert str(app.REPO_ROOT) in __import__('sys').path; print('dashboard import ok')",
        ],
        cwd=tmp_path,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert "dashboard import ok" in result.stdout
