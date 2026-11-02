"""Replay the chronological test split to the Kafka transactions topic."""
from __future__ import annotations

import argparse
import json
import time
from typing import Any, Callable

import pandas as pd
from confluent_kafka import Producer

from src.common.config import (
    KAFKA_BOOTSTRAP_DOCKER,
    KAFKA_BOOTSTRAP_HOST,
    KAFKA_TRANSACTIONS_TOPIC,
    PROCESSED_DIR,
)

FEATURES_PATH = PROCESSED_DIR / "features.parquet"


def ordered_test_rows(frame: pd.DataFrame, limit: int | None = None) -> pd.DataFrame:
    """Return the test split in event-time order, with transaction ID tie-breaks."""
    required = {"split", "transaction_id", "event_timestamp"}
    absent = required.difference(frame.columns)
    if absent:
        raise ValueError(f"Missing producer columns: {sorted(absent)}")
    rows = frame.loc[frame["split"] == "test"].sort_values(
        ["event_timestamp", "transaction_id"], kind="mergesort"
    )
    if limit is not None:
        if limit < 0:
            raise ValueError("limit must be non-negative")
        rows = rows.head(limit)
    return rows.reset_index(drop=True)


def replay_test_split(
    frame: pd.DataFrame,
    producer: Any,
    topic: str = KAFKA_TRANSACTIONS_TOPIC,
    *,
    rate: float | None = None,
    limit: int | None = None,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> int:
    if rate is not None and rate <= 0:
        raise ValueError("rate must be greater than zero")
    rows = ordered_test_rows(frame, limit)
    interval = 0.0 if rate is None else 1.0 / rate
    next_send = clock()
    sent = 0
    for row in rows.itertuples(index=False):
        now = clock()
        delay = next_send - now
        if delay > 0:
            sleep(delay)
        event_timestamp = row.event_timestamp
        payload = {
            "transaction_id": int(row.transaction_id),
            "event_timestamp": event_timestamp.isoformat() if hasattr(event_timestamp, "isoformat") else str(event_timestamp),
        }
        producer.produce(
            topic,
            key=str(payload["transaction_id"]),
            value=json.dumps(payload, separators=(",", ":")),
        )
        producer.poll(0)
        sent += 1
        next_send += interval
    outstanding = producer.flush(30)
    if outstanding:
        raise RuntimeError(f"Kafka producer still has {outstanding} message(s) queued after flush")
    return sent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rate", type=float, default=100.0, help="transactions per second")
    parser.add_argument("--limit", type=int, default=None, help="maximum number of test rows to send")
    parser.add_argument("--docker", action="store_true", help="use the docker-network Kafka address")
    parser.add_argument("--bootstrap-server", default=None)
    parser.add_argument("--topic", default=KAFKA_TRANSACTIONS_TOPIC)
    parser.add_argument("--features", default=str(FEATURES_PATH))
    args = parser.parse_args()
    bootstrap = args.bootstrap_server or (KAFKA_BOOTSTRAP_DOCKER if args.docker else KAFKA_BOOTSTRAP_HOST)
    producer = Producer({"bootstrap.servers": bootstrap, "client.id": "fraud-test-replay"})
    sent = replay_test_split(pd.read_parquet(args.features), producer, args.topic, rate=args.rate, limit=args.limit)
    print(json.dumps({"bootstrap_server": bootstrap, "topic": args.topic, "published": sent}))


if __name__ == "__main__":
    main()
