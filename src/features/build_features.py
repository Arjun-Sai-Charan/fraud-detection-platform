"""Build strictly causal transaction features and a persisted category encoder."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.config import PROCESSED_DIR, RESULTS_DIR
from src.features.categorical import (
    CATEGORICAL_COLUMNS,
    DEFAULT_ENCODER_PATH,
    fit_encoder,
    transform_categories,
)

IEEE_CIS_EPOCH = pd.Timestamp("2017-11-30T00:00:00Z")
RAW_CATEGORICAL_COLUMNS = CATEGORICAL_COLUMNS
RAW_NUMERIC_COLUMNS = tuple(f"C{i}" for i in range(1, 15)) + tuple(f"D{i}" for i in range(1, 16))
RAW_COLUMNS = ("TransactionAmt",) + RAW_CATEGORICAL_COLUMNS + RAW_NUMERIC_COLUMNS


def _same_value(left: object, right: object) -> bool:
    left_missing, right_missing = pd.isna(left), pd.isna(right)
    if left_missing and right_missing:
        return True
    if left_missing or right_missing:
        return False
    return bool(left == right)


def engineer_causal_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Compute per-uid features using only rows at strictly earlier timestamps.

    The 1h and 24h intervals include their left boundary and exclude the event
    timestamp. Transactions with tied timestamps are evaluated as one batch, so
    none of the tied rows can observe another tied row.
    """
    required = {"TransactionID", "TransactionDT", "uid", "TransactionAmt", "DeviceInfo", "addr1"}
    absent = required.difference(frame.columns)
    if absent:
        raise ValueError(f"Missing columns needed for causal features: {sorted(absent)}")
    if frame["uid"].isna().any():
        raise ValueError("uid cannot be null")

    ordered = frame.sort_values(["TransactionDT", "TransactionID"], kind="mergesort").reset_index(drop=True).copy()
    if ordered["TransactionDT"].isna().any():
        raise ValueError("TransactionDT cannot be null")
    transaction_dt = ordered["TransactionDT"].to_numpy(dtype=np.int64)
    amounts = pd.to_numeric(ordered["TransactionAmt"], errors="coerce").to_numpy(dtype=np.float64)
    n = len(ordered)

    count_1h = np.zeros(n, dtype=np.int64)
    count_24h = np.zeros(n, dtype=np.int64)
    prior_count = np.zeros(n, dtype=np.int64)
    prior_mean = np.full(n, np.nan, dtype=np.float64)
    device_change = np.full(n, np.nan, dtype=np.float64)
    location_change = np.full(n, np.nan, dtype=np.float64)
    is_first = np.zeros(n, dtype=np.int8)
    is_hub = np.zeros(n, dtype=np.int8)

    for indices in ordered.groupby("uid", sort=False).indices.values():
        idx = np.asarray(indices, dtype=np.int64)
        uid_times = transaction_dt[idx]
        uid_amounts = amounts[idx]
        uid_devices = ordered["DeviceInfo"].iloc[idx].to_numpy(dtype=object)
        uid_locations = ordered["addr1"].iloc[idx].to_numpy(dtype=object)
        starts = np.r_[0, np.flatnonzero(uid_times[1:] != uid_times[:-1]) + 1]
        ends = np.r_[starts[1:], len(idx)]
        finite_amount = np.isfinite(uid_amounts)
        prefix_amount = np.r_[0.0, np.where(finite_amount, uid_amounts, 0.0).cumsum()]
        prefix_valid = np.r_[0, finite_amount.astype(np.int64).cumsum()]

        for bucket_start, bucket_end in zip(starts, ends):
            event_time = uid_times[bucket_start]
            prior = int(bucket_start)
            global_rows = idx[bucket_start:bucket_end]
            prior_count[global_rows] = prior
            is_first[global_rows] = int(prior == 0)
            is_hub[global_rows] = int(prior > 1000)
            first_1h = np.searchsorted(uid_times[:bucket_start], event_time - 3600, side="left")
            first_24h = np.searchsorted(uid_times[:bucket_start], event_time - 86400, side="left")
            count_1h[global_rows] = prior - first_1h
            count_24h[global_rows] = prior - first_24h
            valid_past = int(prefix_valid[bucket_start])
            if valid_past:
                mean = prefix_amount[bucket_start] / valid_past
                prior_mean[global_rows] = mean
            if bucket_start:
                previous_device = uid_devices[bucket_start - 1]
                previous_location = uid_locations[bucket_start - 1]
                device_change[global_rows] = [not _same_value(value, previous_device) for value in uid_devices[bucket_start:bucket_end]]
                location_change[global_rows] = [not _same_value(value, previous_location) for value in uid_locations[bucket_start:bucket_end]]

    ordered["txn_count_last_1h"] = count_1h
    ordered["txn_count_last_24h"] = count_24h
    ordered["amount_to_past_mean"] = amounts / prior_mean
    ordered["device_change_vs_previous"] = device_change
    ordered["location_change_vs_previous"] = location_change
    ordered["is_first_txn_for_uid"] = is_first
    ordered["uid_prior_txn_count"] = prior_count
    ordered["uid_is_hub"] = is_hub

    # Keep the raw modeling fields named in the phase spec; V* fields are not used.
    present_raw = [column for column in RAW_COLUMNS if column in ordered.columns]
    result = ordered.rename(columns={"TransactionID": "transaction_id"}).copy()
    result["transaction_dt"] = transaction_dt
    result["event_timestamp"] = IEEE_CIS_EPOCH + pd.to_timedelta(transaction_dt, unit="s")
    keep = [
        "transaction_id",
        "event_timestamp",
        "transaction_dt",
        "uid",
        *present_raw,
        "txn_count_last_1h",
        "txn_count_last_24h",
        "amount_to_past_mean",
        "device_change_vs_previous",
        "location_change_vs_previous",
        "is_first_txn_for_uid",
        "uid_prior_txn_count",
        "uid_is_hub",
        "isFraud",
        "split",
    ]
    return result.loc[:, [column for column in keep if column in result.columns]]


def load_phase1_rows(processed_dir: Path = PROCESSED_DIR) -> pd.DataFrame:
    """Read existing Phase 1 parquet splits without changing them."""
    parts = []
    for name in ("train", "validation", "test"):
        part = pd.read_parquet(processed_dir / f"{name}.parquet")
        part["split"] = name
        parts.append(part)
    return pd.concat(parts, ignore_index=True).sort_values(
        ["TransactionDT", "TransactionID"], kind="mergesort", ignore_index=True
    )


def build_feature_table(
    input_frame: pd.DataFrame,
    output_path: Path = PROCESSED_DIR / "features.parquet",
    encoder_path: Path = DEFAULT_ENCODER_PATH,
) -> pd.DataFrame:
    """Engineer features, fit categorical vocabulary on train, and save artifacts."""
    features = engineer_causal_features(input_frame)
    categorical_columns = tuple(column for column in CATEGORICAL_COLUMNS if column in features.columns)
    if categorical_columns:
        train_mask = input_frame.sort_values(["TransactionDT", "TransactionID"], kind="mergesort")["split"].eq("train").to_numpy()
        # engineer_causal_features sorted the same way, so this mask aligns.
        fit_columns = categorical_columns
        artifact = fit_encoder(features.loc[train_mask], encoder_path, fit_columns)
        encoded = transform_categories(features, artifact)
        for column in categorical_columns:
            features[column] = encoded[column]
            features[f"{column}_encoded"] = encoded[f"{column}_encoded"].astype(np.float64)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(output_path, index=False)
    return features


def write_feature_summary(features: pd.DataFrame, path: Path = RESULTS_DIR / "features_summary.json") -> dict[str, object]:
    """Save the actual feature-table dimensions and time/split coverage."""
    summary: dict[str, object] = {
        "rows": int(len(features)),
        "columns": int(features.shape[1]),
        "feature_columns": list(features.columns),
        "split_rows": {str(name): int(count) for name, count in features["split"].value_counts().items()},
        "event_timestamp_range_utc": {
            "min": features["event_timestamp"].min().isoformat(),
            "max": features["event_timestamp"].max().isoformat(),
        },
        "categorical_encoder_columns": list(CATEGORICAL_COLUMNS),
        "categorical_encoder_fit_split": "train",
        "hub_flag_rule": "uid_prior_txn_count > 1000, based only on strictly earlier timestamps",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--output", type=Path, default=PROCESSED_DIR / "features.parquet")
    parser.add_argument("--encoder", type=Path, default=DEFAULT_ENCODER_PATH)
    args = parser.parse_args()
    source = load_phase1_rows(args.input_dir)
    result = build_feature_table(source, args.output, args.encoder)
    summary = write_feature_summary(result)
    print(f"Wrote {len(result):,} causal feature rows to {args.output}")
    print(f"Categorical encoding artifact: {args.encoder}")
    print(f"Feature summary: {summary['rows']:,} rows x {summary['columns']} columns")


if __name__ == "__main__":
    main()
