"""Compare offline fraud rates inside and outside flagged test rings."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

import mlflow.xgboost
import numpy as np
import pandas as pd

from src.common.config import GNN_ARTIFACT_DIR, RESULTS_DIR, XGB_ARTIFACT_DIR

FEATURES_PATH = Path("data/processed/features.parquet")
OUTPUT_PATH = RESULTS_DIR / "ring_precision.json"
ENTITY_COLUMNS = {
    "uid": "uid",
    "device": "DeviceInfo",
    "email_domain": "P_emaildomain",
    "addr1": "addr1",
}
STRONG_ENTITY_TYPES = {"uid", "device"}
WEAK_ENTITY_TYPES = {"email_domain", "addr1"}


def _entity_key(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return None if text.lower() in {"", "nan", "none", "<na>", "null"} else text


def load_non_hub_entity_keys(
    split_paths: dict[str, Path],
    hub_report_path: Path,
) -> dict[str, set[str]]:
    """Recreate the Phase 5 all-snapshot hub filter without changing its artifact."""
    hub_report = json.loads(hub_report_path.read_text(encoding="utf-8"))
    non_hub: dict[str, set[str]] = {}
    for entity_type, column in ENTITY_COLUMNS.items():
        degree_counts: Counter[str] = Counter()
        for path in split_paths.values():
            values = pd.read_parquet(path, columns=[column])[column]
            degree_counts.update(key for key in map(_entity_key, values) if key is not None)
        degree_cap = int(hub_report[entity_type]["degree_cap_from_train_percentile"])
        non_hub[entity_type] = {
            key for key, degree in degree_counts.items() if degree <= degree_cap
        }
    return non_hub


def analyze_ring_links(
    flagged_rows: pd.DataFrame,
    entity_rows: pd.DataFrame,
    non_hub_entities: dict[str, set[str]],
) -> dict[str, Any]:
    """Classify each existing ring by the actual shared non-hub entity links."""
    joined = flagged_rows[["transaction_id", "isFraud", "ring_id", "ring_size"]].merge(
        entity_rows[["transaction_id", *ENTITY_COLUMNS.values()]],
        on="transaction_id",
        how="left",
        validate="one_to_one",
    )
    ring_types: dict[str, str] = {}
    for ring_id, rows in joined.loc[joined["ring_id"].notna()].groupby("ring_id", sort=False):
        has_shared: set[str] = set()
        for entity_type, column in ENTITY_COLUMNS.items():
            allowed = non_hub_entities.get(entity_type, set())
            counts = Counter(
                key for key in map(_entity_key, rows[column])
                if key is not None and key in allowed
            )
            if any(count >= 2 for count in counts.values()):
                has_shared.add(entity_type)
        has_strong = bool(has_shared & STRONG_ENTITY_TYPES)
        has_weak = bool(has_shared & WEAK_ENTITY_TYPES)
        if has_weak:
            ring_types[str(ring_id)] = "includes_weak_link" if has_strong else "weak_only"
        elif has_strong:
            ring_types[str(ring_id)] = "strong_only"
        else:
            ring_types[str(ring_id)] = "unclassified"

    joined["ring_link_type"] = joined["ring_id"].map(
        lambda value: ring_types.get(str(value), "outside_ring") if pd.notna(value) else "outside_ring"
    )

    def fraud_summary(rows: pd.DataFrame) -> dict[str, Any]:
        count = int(len(rows))
        fraud_count = int(rows["isFraud"].sum())
        return {
            "transaction_count": count,
            "fraud_count": fraud_count,
            "fraud_rate": float(fraud_count / count) if count else 0.0,
        }

    ring_sizes = joined.loc[joined["ring_id"].notna()].groupby("ring_id").agg(
        ring_size=("ring_size", "max"), flagged_rows=("transaction_id", "count")
    )
    if ring_sizes.empty:
        size_summary = {
            "ring_count": 0,
            "flagged_rows_in_rings": 0,
            "median": None,
            "p90": None,
            "max": None,
            "largest_ring_id": None,
            "largest_ring_flagged_rows": 0,
            "share_of_all_flagged_rows_in_largest_ring": 0.0,
            "ring_count_by_size": {},
        }
    else:
        sizes = pd.to_numeric(ring_sizes["ring_size"], errors="coerce").dropna()
        largest_index = ring_sizes["flagged_rows"].idxmax()
        size_summary = {
            "ring_count": int(len(ring_sizes)),
            "flagged_rows_in_rings": int(ring_sizes["flagged_rows"].sum()),
            "median": float(sizes.median()) if len(sizes) else None,
            "p90": float(sizes.quantile(0.9)) if len(sizes) else None,
            "max": int(sizes.max()) if len(sizes) else None,
            "largest_ring_id": str(largest_index),
            "largest_ring_flagged_rows": int(ring_sizes.loc[largest_index, "flagged_rows"]),
            "share_of_all_flagged_rows_in_largest_ring": float(
                ring_sizes.loc[largest_index, "flagged_rows"] / len(joined)
            ) if len(joined) else 0.0,
            "ring_count_by_size": {
                str(int(size)): int(count)
                for size, count in ring_sizes.groupby("ring_size", dropna=True).size().items()
            },
        }
    return {
        "ring_size_distribution": size_summary,
        "fraud_rate_by_link_type": {
            "strong_links_only": fraud_summary(joined.loc[joined["ring_link_type"].eq("strong_only")]),
            "includes_weak_link": fraud_summary(joined.loc[joined["ring_link_type"].eq("includes_weak_link") | joined["ring_link_type"].eq("weak_only")]),
            "outside_any_ring": fraud_summary(joined.loc[joined["ring_link_type"].eq("outside_ring")]),
        },
        "ring_counts_by_link_type": {
            "strong_links_only": sum(value == "strong_only" for value in ring_types.values()),
            "includes_weak_link": sum(value in {"includes_weak_link", "weak_only"} for value in ring_types.values()),
            "only_weak_links": sum(value == "weak_only" for value in ring_types.values()),
            "unclassified": sum(value == "unclassified" for value in ring_types.values()),
        },
    }


def compare_ring_precision(
    features: pd.DataFrame,
    model: Any,
    feature_names: list[str],
    rings: pd.DataFrame,
    blend_config: dict[str, Any],
    entity_rows: pd.DataFrame | None = None,
    non_hub_entities: dict[str, set[str]] | None = None,
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

    result = {
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
    if entity_rows is not None and non_hub_entities is not None:
        result["ring_definition_breakdown"] = analyze_ring_links(
            flagged_rows, entity_rows, non_hub_entities
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=FEATURES_PATH)
    parser.add_argument("--rings", type=Path, default=GNN_ARTIFACT_DIR / "rings.parquet")
    parser.add_argument("--hub-report", type=Path, default=GNN_ARTIFACT_DIR / "hub_report.json")
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()
    features = pd.read_parquet(args.features)
    feature_payload = json.loads((XGB_ARTIFACT_DIR / "feature_list.json").read_text(encoding="utf-8"))
    model = mlflow.xgboost.load_model(str(XGB_ARTIFACT_DIR / "model"))
    blend_config = json.loads((GNN_ARTIFACT_DIR / "blend_config.json").read_text(encoding="utf-8"))
    rings = pd.read_parquet(args.rings)
    split_paths = {
        split: args.processed_dir / f"{split}.parquet"
        for split in ("train", "validation", "test")
    }
    non_hub_entities = load_non_hub_entity_keys(split_paths, args.hub_report)
    entity_rows = pd.read_parquet(
        args.processed_dir / "test.parquet",
        columns=["TransactionID", *ENTITY_COLUMNS.values()],
    ).rename(columns={"TransactionID": "transaction_id"})
    result = compare_ring_precision(
        features,
        model,
        feature_payload["features"],
        rings,
        blend_config,
        entity_rows,
        non_hub_entities,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
