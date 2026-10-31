"""Compare Feast historical and SQLite online values for the same 100 rows."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from feast import FeatureStore

from src.common.config import PROCESSED_DIR, RESULTS_DIR

REPO_PATH = Path(__file__).resolve().parents[1] / "feature_repo"
FEATURES_PATH = PROCESSED_DIR / "features.parquet"
PARITY_PATH = RESULTS_DIR / "feature_parity.json"


def _equal_values(left: pd.Series, right: pd.Series) -> bool:
    left_missing = left.isna()
    right_missing = right.isna()
    if not left_missing.equals(right_missing):
        return False
    valid = ~left_missing
    if not valid.any():
        return True
    lvals, rvals = left[valid], right[valid]
    if pd.api.types.is_numeric_dtype(lvals) and pd.api.types.is_numeric_dtype(rvals):
        return bool(np.allclose(lvals.astype(float), rvals.astype(float), rtol=1e-6, atol=1e-8))
    return all(a == b for a, b in zip(lvals.tolist(), rvals.tolist()))


def run_parity_check(sample_size: int = 100) -> dict[str, object]:
    data = pd.read_parquet(FEATURES_PATH)
    sample = data.sample(n=sample_size, random_state=42).sort_values("transaction_id")
    entity_rows = sample[["transaction_id"]].astype({"transaction_id": "int64"}).to_dict("records")
    entity_df = sample[["transaction_id", "event_timestamp"]].copy()
    entity_df["transaction_id"] = entity_df["transaction_id"].astype("int64")

    store = FeatureStore(repo_path=str(REPO_PATH))
    feature_names = [
        field.name
        for field in store.get_feature_view("transaction_features").schema
        if field.name != "transaction_id"
    ]
    feature_refs = [f"transaction_features:{name}" for name in feature_names]
    historical = store.get_historical_features(
        entity_df=entity_df,
        features=feature_refs,
        full_feature_names=True,
    ).to_df()
    online = store.get_online_features(
        features=feature_refs,
        entity_rows=entity_rows,
        full_feature_names=True,
    ).to_df()

    historical = historical.sort_values("transaction_id").reset_index(drop=True)
    if "transaction_id" not in online:
        online.insert(0, "transaction_id", sample["transaction_id"].to_numpy(dtype=np.int64))
    online = online.sort_values("transaction_id").reset_index(drop=True)
    mismatches = []
    for name in feature_names:
        field = f"transaction_features__{name}"
        if field not in historical:
            field = f"transaction_features:{name}"
        if field not in online:
            field = f"transaction_features:{name}"
        if field not in historical or field not in online or not _equal_values(historical[field], online[field]):
            mismatches.append(name)
    result = {
        "sample_size": int(sample_size),
        "features_checked": int(len(feature_names)),
        "equal_features": int(len(feature_names) - len(mismatches)),
        "mismatched_features": mismatches,
        "parity_passed": not mismatches,
    }
    PARITY_PATH.parent.mkdir(parents=True, exist_ok=True)
    PARITY_PATH.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if mismatches:
        raise AssertionError(f"Feast offline/online parity failed for: {mismatches}")
    return result


if __name__ == "__main__":
    print(json.dumps(run_parity_check(), indent=2))
