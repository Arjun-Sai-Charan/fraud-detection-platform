"""Feast definitions for transaction features (Feast 0.40.1 API)."""
from datetime import timedelta
from pathlib import Path

from feast import Entity, FeatureView, Field, FileSource, ValueType
from feast.types import Float64, Int64, String

ROOT = Path(__file__).resolve().parents[2]
FEATURES_PATH = ROOT / "data" / "processed" / "features.parquet"

transaction = Entity(
    name="transaction_id",
    join_keys=["transaction_id"],
    value_type=ValueType.INT64,
    description="Unique transaction identifier",
)

source = FileSource(
    path=str(FEATURES_PATH),
    timestamp_field="event_timestamp",
    description="Chronological causal features from the Phase 1 training window",
)

_fields = [
    Field(name="uid", dtype=String),
    Field(name="TransactionAmt", dtype=Float64),
    Field(name="txn_count_last_1h", dtype=Int64),
    Field(name="txn_count_last_24h", dtype=Int64),
    Field(name="amount_to_past_mean", dtype=Float64),
    Field(name="device_change_vs_previous", dtype=Float64),
    Field(name="location_change_vs_previous", dtype=Float64),
    Field(name="is_first_txn_for_uid", dtype=Int64),
    Field(name="uid_prior_txn_count", dtype=Int64),
    Field(name="uid_is_hub", dtype=Int64),
]
_fields += [
    Field(name=column, dtype=String)
    for column in ("ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain", *(f"M{i}" for i in range(1, 10)))
]
_fields += [
    Field(name=column, dtype=Float64)
    for column in ("C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10", "C11", "C12", "C13", "C14", *(f"D{i}" for i in range(1, 16)))
]
_fields += [
    Field(name=f"{column}_encoded", dtype=Float64)
    for column in ("ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain", *(f"M{i}" for i in range(1, 10)))
]

transaction_features = FeatureView(
    name="transaction_features",
    entities=[transaction],
    schema=_fields,
    source=source,
    ttl=timedelta(days=3650),
    description="Causal per-transaction features; labels and split metadata are excluded.",
)
