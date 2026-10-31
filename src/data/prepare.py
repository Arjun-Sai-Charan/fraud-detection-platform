"""Create a time-ordered, time-split parquet dataset from the training CSVs."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from src.common.config import (
    IDENTITY_FILE,
    PROCESSED_DIR,
    RAW_DIR,
    RESULTS_DIR,
    SUBSAMPLE_ROWS,
    TRAIN_FRACTION,
    TRANSACTION_FILE,
    VALIDATION_FRACTION,
)

UID_COLUMNS = ("card1", "card2", "card3", "card5", "addr1")
UID_SENTINEL = "<MISSING>"


def _uid_part(value: Any) -> str:
    if pd.isna(value):
        return UID_SENTINEL
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def build_uid(frame: pd.DataFrame) -> pd.Series:
    """Return stable SHA-256 identifiers for the configured card/address tuple."""
    absent = set(UID_COLUMNS).difference(frame.columns)
    if absent:
        raise ValueError(f"Missing UID columns: {sorted(absent)}")

    def digest(row: tuple[Any, ...]) -> str:
        payload = json.dumps([_uid_part(value) for value in row], ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    return pd.Series(
        (digest(row) for row in frame.loc[:, UID_COLUMNS].itertuples(index=False, name=None)),
        index=frame.index,
        name="uid",
        dtype="string",
    )


def load_and_join(
    transaction_path: Path = TRANSACTION_FILE,
    identity_path: Path = IDENTITY_FILE,
) -> pd.DataFrame:
    """Select the earliest rows and left join identity without loading all columns at once.

    A narrow first pass finds the earliest source row positions. Chunked second
    passes load just those transaction rows and matching identity rows, keeping
    peak memory reasonable for the wide IEEE-CIS CSVs.
    """
    metadata = pd.read_csv(transaction_path, usecols=["TransactionID", "TransactionDT"])
    if metadata["TransactionID"].duplicated().any():
        raise ValueError("Transaction table contains duplicate TransactionID values")
    selected_positions = metadata["TransactionDT"].sort_values(kind="mergesort").head(SUBSAMPLE_ROWS).index
    selected_positions = set(int(position) for position in selected_positions)
    selected_ids = set(metadata.loc[list(selected_positions), "TransactionID"].tolist())

    transaction_parts = []
    offset = 0
    for chunk in pd.read_csv(transaction_path, chunksize=25_000):
        local_positions = [position - offset for position in selected_positions if offset <= position < offset + len(chunk)]
        if local_positions:
            transaction_parts.append(chunk.iloc[local_positions])
        offset += len(chunk)
    transactions = pd.concat(transaction_parts, ignore_index=True)
    if transactions["TransactionID"].duplicated().any():
        raise ValueError("Selected transactions contain duplicate TransactionID values")

    identity_parts = []
    for chunk in pd.read_csv(identity_path, chunksize=25_000):
        matches = chunk[chunk["TransactionID"].isin(selected_ids)]
        if not matches.empty:
            identity_parts.append(matches)
    identity = pd.concat(identity_parts, ignore_index=True) if identity_parts else pd.DataFrame(columns=["TransactionID"])
    if identity["TransactionID"].duplicated().any():
        raise ValueError("Identity table contains duplicate TransactionID values")
    merged = transactions.merge(identity, how="left", on="TransactionID", validate="one_to_one", sort=False)
    return merged.sort_values("TransactionDT", kind="mergesort", ignore_index=True)


def split_ordered(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Take the first configured time window and split it without shuffling."""
    if "TransactionDT" not in frame or "isFraud" not in frame:
        raise ValueError("Input must contain TransactionDT and isFraud")
    ordered = frame.sort_values("TransactionDT", kind="mergesort").head(SUBSAMPLE_ROWS).reset_index(drop=True).copy()
    ordered["uid"] = build_uid(ordered)
    count = len(ordered)
    train_end = math.floor(count * TRAIN_FRACTION)
    validation_end = train_end + math.floor(count * VALIDATION_FRACTION)
    return {
        "train": ordered.iloc[:train_end].reset_index(drop=True),
        "validation": ordered.iloc[train_end:validation_end].reset_index(drop=True),
        "test": ordered.iloc[validation_end:].reset_index(drop=True),
    }


def _uid_distribution(frame: pd.DataFrame) -> dict[str, float | int]:
    per_uid = frame.groupby("uid", sort=False).size()
    return {
        "unique_uids": int(per_uid.size),
        "transactions_per_uid": {
            "min": int(per_uid.min()),
            "median": float(per_uid.median()),
            "mean": float(per_uid.mean()),
            "p90": float(per_uid.quantile(0.90)),
            "p95": float(per_uid.quantile(0.95)),
            "max": int(per_uid.max()),
        },
    }


def build_summary(splits: dict[str, pd.DataFrame]) -> dict[str, Any]:
    combined = pd.concat(splits.values(), ignore_index=True)
    boundaries = {
        name: {
            "rows": int(len(part)),
            "min_transaction_dt": int(part["TransactionDT"].min()),
            "max_transaction_dt": int(part["TransactionDT"].max()),
        }
        for name, part in splits.items()
    }
    uid_stats = _uid_distribution(combined)
    return {
        "rows": int(len(combined)),
        "columns": int(combined.shape[1]),
        "column_names": list(combined.columns),
        "time_range_transaction_dt": {
            "min": int(combined["TransactionDT"].min()),
            "max": int(combined["TransactionDT"].max()),
        },
        "fraud_rate_overall": float(combined["isFraud"].mean()),
        "fraud_rate_per_split": {name: float(part["isFraud"].mean()) for name, part in splits.items()},
        "unique_uids": uid_stats["unique_uids"],
        "transactions_per_uid": uid_stats["transactions_per_uid"],
        "split_boundaries": boundaries,
    }


def prepare_data(
    transaction_path: Path = TRANSACTION_FILE,
    identity_path: Path = IDENTITY_FILE,
    output_dir: Path = PROCESSED_DIR,
    summary_path: Path = RESULTS_DIR / "eda_summary.json",
) -> dict[str, Any]:
    """Prepare the first chronological window, save its splits, and report stats."""
    merged = load_and_join(transaction_path, identity_path)
    splits = split_ordered(merged)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, part in splits.items():
        part.to_parquet(output_dir / f"{name}.parquet", index=False)
    summary = build_summary(splits)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transactions", type=Path, default=TRANSACTION_FILE)
    parser.add_argument("--identity", type=Path, default=IDENTITY_FILE)
    parser.add_argument("--output-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--summary", type=Path, default=RESULTS_DIR / "eda_summary.json")
    args = parser.parse_args()
    print(json.dumps(prepare_data(args.transactions, args.identity, args.output_dir, args.summary), indent=2))


if __name__ == "__main__":
    main()
