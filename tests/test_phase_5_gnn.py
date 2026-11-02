import numpy as np
import pandas as pd

from src.models.blend import blend_scores, connected_rings
from src.models.build_graph import cap_entity_edges, normalize_train_only
from src.serving.scorer import FraudScorer


def test_blend_math():
    result = blend_scores(np.array([0.2, 0.8]), np.array([0.6, 0.4]), 0.75)
    np.testing.assert_allclose(result, [0.3, 0.7])


def test_ring_components_join_on_any_non_hub_entity():
    rows = connected_rings(
        np.array([10, 11, 12, 13]),
        np.array([True, True, True, True]),
        {
            "uid": ["u1", "u2", "u2", "u4"],
            "device": ["d1", "d1", "d2", "d4"],
            "email_domain": [None, None, "e2", None],
            "addr1": [None, None, None, "a4"],
        },
        {
            "uid": {"u1", "u2", "u4"},
            "device": {"d1", "d2", "d4"},
            "email_domain": {"e2"},
            "addr1": {"a4"},
        },
    )
    assert [row["ring_size"] for row in rows] == [3, 3, 3, None]
    assert rows[0]["ring_id"] == rows[2]["ring_id"] == "ring-10"
    assert rows[3]["ring_id"] is None


def test_graph_builder_hub_cap_drops_all_edges_to_over_limit_entity():
    tx, entity, keep = cap_entity_edges(
        np.array([0, 1, 2, 3, 4, 5]),
        np.array([0, 0, 0, 1, 1, 2]),
        max_degree=2,
    )
    assert keep.tolist() == [False, False, False, True, True, True]
    assert tx.tolist() == [3, 4, 5]
    assert entity.tolist() == [1, 1, 2]


def test_transaction_normalization_uses_train_rows_only():
    values = np.array([[1.0, 10.0], [3.0, 10.0], [1000.0, np.nan]])
    normalized, stats = normalize_train_only(values, train_count=2)
    assert stats["mean"] == [2.0, 10.0]
    assert stats["std"] == [1.0, 1.0]
    np.testing.assert_allclose(normalized[:2, 0], [-1.0, 1.0])
    assert normalized[2, 0] == 998.0
    assert normalized[2, 1] == 0.0


def test_scorer_falls_back_to_xgb_when_gnn_artifacts_are_absent(tmp_path):
    class Store:
        def get_online_features(self, **_kwargs):
            class Response:
                def to_dict(self):
                    return {"transaction_features__amount": [12.0]}
            return Response()

    class Model:
        def predict_proba(self, _frame):
            return np.array([[0.2, 0.8]])

    scorer = FraudScorer(
        model=Model(),
        feature_store=Store(),
        feature_list=["amount"],
        threshold={"threshold": 0.5, "tie_break_transaction_id_max": -1},
        gnn_artifact_dir=tmp_path / "missing-gnn",
    )
    result = scorer.score(999)
    assert result["xgb_score"] == result["final_score"] == 0.8
    assert result["gnn_score"] is None
    assert result["alert"] is True
    assert result["ring_id"] is None and result["ring_size"] is None


def test_scorer_uses_precomputed_gnn_blend_and_ring(tmp_path):
    import json

    class Store:
        def get_online_features(self, **_kwargs):
            class Response:
                def to_dict(self):
                    return {"transaction_features__amount": [12.0]}
            return Response()

    class Model:
        def predict_proba(self, _frame):
            return np.array([[0.2, 0.8]])

    pd.DataFrame([{"transaction_id": 1001, "gnn_score": 0.2}]).to_parquet(tmp_path / "gnn_scores.parquet", index=False)
    pd.DataFrame([{"transaction_id": 1001, "ring_id": "ring-1000", "ring_size": 4}]).to_parquet(tmp_path / "rings.parquet", index=False)
    (tmp_path / "blend_config.json").write_text(json.dumps({
        "blend_weight_xgb": 0.25,
        "threshold": 0.3,
        "tie_break_transaction_id_max": 1000,
    }))
    scorer = FraudScorer(
        model=Model(),
        feature_store=Store(),
        feature_list=["amount"],
        threshold={"threshold": 0.7, "tie_break_transaction_id_max": -1},
        gnn_artifact_dir=tmp_path,
    )
    result = scorer.score(1001)
    assert result["gnn_score"] == 0.2
    assert np.isclose(result["final_score"], 0.35)
    assert result["alert"] is True
    assert result["ring_id"] == "ring-1000" and result["ring_size"] == 4
