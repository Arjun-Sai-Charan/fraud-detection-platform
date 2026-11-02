"""Benchmark Feast online lookup plus XGBoost inference on 5,000 test rows."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.config import PROCESSED_DIR, RESULTS_DIR
from src.serving.scorer import FraudScorer

FEATURES_PATH = PROCESSED_DIR / "features.parquet"
OUTPUT_PATH = RESULTS_DIR / "latency.json"


def benchmark(scorer: FraudScorer, features_path: Path = FEATURES_PATH, count: int = 5000) -> dict[str, float | int | str]:
    rows = pd.read_parquet(features_path, columns=["transaction_id", "event_timestamp", "split"])
    rows = rows.loc[rows["split"] == "test"].sort_values(
        ["event_timestamp", "transaction_id"], kind="mergesort"
    ).head(count)
    if len(rows) < count:
        raise ValueError(f"Requested {count} test transactions, found {len(rows)}")
    scorer._ensure_ready()
    latencies = []
    started = time.perf_counter()
    for transaction_id in rows["transaction_id"].astype(int):
        response = scorer.score(transaction_id)
        latencies.append(float(response["latency_ms"]))
    elapsed = time.perf_counter() - started
    p50, p95, p99 = np.percentile(np.asarray(latencies), [50, 95, 99]).tolist()
    return {
        "transactions": int(len(latencies)),
        "latency_scope": "Feast online lookup plus model inference; excludes Kafka and model startup",
        "p50_ms": float(p50),
        "p95_ms": float(p95),
        "p99_ms": float(p99),
        "throughput_transactions_per_second": float(len(latencies) / elapsed),
        "elapsed_seconds": float(elapsed),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=FEATURES_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--count", type=int, default=5000)
    parser.add_argument("--model-source", choices=("local", "registry"), default="local")
    parser.add_argument("--tracking-uri", default=None)
    args = parser.parse_args()
    result = benchmark(FraudScorer(model_source=args.model_source, tracking_uri=args.tracking_uri), args.features, args.count)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
