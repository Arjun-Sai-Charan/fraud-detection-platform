"""Compare offline fraud rates inside and outside flagged test rings."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import mlflow.xgboost
import numpy as np
import pandas as pd

from src.common.config import GNN_ARTIFACT_DIR, RESULTS_DIR, XGB_ARTIFACT_DIR

FEATURES_PATH = Path("data/processed/features.parquet")
OUTPUT_PATH = RESULTS_DIR / "ring_precision.json"


def compare_ring_precision(
    features: pd.DataFrame,
    model: Any,
    feature_names: list[str],
    rings: pd.DataFrame,
    blend_config: dict[str, Any],
) -> dict[str, Any]:
    test = features.loc[features["split"].eq("test")].copy()
    scores = np.asarray(model.predict_proba(test[feature_names])[:, 1], dtype=np.float64)
    weight = float(blend_config["blend_weight_xgb"])
    gnn = pd.read_parquet(GNN_ARTIFACT_DIR / "gnn_scores.parquet", columns=["transaction_id", "split", "gnn_score"])
    gnn = gnn.loc[gnn["split"].eq("test"), ["transaction_id", "gnn_score"]]
    test = test.merge(gnn, on="transaction_id", how="left", validate="one_to_one")
    gnn_scores = test["gnn_score"].fillna(0.0).to_numpy(dtype=np.float64)
    final_scores = weight * scores + (1.0 - weight) * gnn_scores
    threshold = float(blend_config["threshold"])
    tie_id = int(blend_config.get("tie_break_transaction_id_max", -1))
    flagged = (final_scores > threshold) | ((final_scores == threshold) & test["transaction_id"].le(tie_id).to_numpy())
    test["blend_score"] = final_scores
    test["flagged"] = flagged
    test = test.merge(rings[["transaction_id", "ring_id", "ring_size"]], on="transaction_id", how="left", validate="one_to_one")
    flagged_rows = test.loc[test["flagged"]].copy()
    inside = flagged_rows["ring_id"].notna() & pd.to_numeric(flagged_rows["ring_size"], errors="coerce").gt(1)

    def summary(rows: pd.DataFrame) -> dict[str, Any]:
        fraud_count = int(rows["isFraud"].sum())
        count = int(len(rows))
        return {
            "transaction_count": count,
            "fraud_count": fraud_count,
            "fraud_rate": float(fraud_count / count) if count else 0.0,
        }

    return {
        "split": "test",
        "flag_rule": "blend score above saved validation top-3% threshold, including saved transaction-id tie rule",
        "blend_weight_xgb": weight,
        "threshold": threshold,
        "flagged_transaction_count": int(len(flagged_rows)),
        "comparison": {
            "inside_multi_transaction_rings": summary(flagged_rows.loc[inside]),
            "outside_multi_transaction_rings": summary(flagged_rows.loc[~inside]),
        },
        "interpretation_note": "Uses ground-truth test labels offline; labels are unavailable during live scoring.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=FEATURES_PATH)
    parser.add_argument("--rings", type=Path, default=GNN_ARTIFACT_DIR / "rings.parquet")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()
    features = pd.read_parquet(args.features)
    feature_payload = json.loads((XGB_ARTIFACT_DIR / "feature_list.json").read_text(encoding="utf-8"))
    model = mlflow.xgboost.load_model(str(XGB_ARTIFACT_DIR / "model"))
    blend_config = json.loads((GNN_ARTIFACT_DIR / "blend_config.json").read_text(encoding="utf-8"))
    rings = pd.read_parquet(args.rings)
    result = compare_ring_precision(features, model, feature_payload["features"], rings, blend_config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
