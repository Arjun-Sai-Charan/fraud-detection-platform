"""Score a chronological sample of test transactions into the demo SQLite DB."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.common.config import PROCESSED_DIR, SERVING_DB_PATH
from src.serving.repository import ScoreRepository
from src.serving.scorer import FraudScorer


def seed_demo_alerts(
    scorer: Any,
    repository: ScoreRepository,
    features_path: Path = PROCESSED_DIR / "features.parquet",
    limit: int = 300,
) -> dict[str, int]:
    rows = pd.read_parquet(features_path, columns=["transaction_id", "event_timestamp", "split"])
    test = rows.loc[rows["split"].eq("test")].sort_values(
        ["event_timestamp", "transaction_id"], kind="mergesort"
    ).reset_index(drop=True)
    if test.empty:
        return {"scored": 0, "alerts": 0}
    count = min(max(0, int(limit)), len(test))
    offsets = np.linspace(0, len(test) - 1, count, dtype=int) if count else np.array([], dtype=int)
    alert_count = 0
    for transaction_id in test.iloc[offsets]["transaction_id"].astype(int):
        result = repository.save_score(scorer.score(transaction_id))
        alert_count += int(bool(result["alert"]))
    return {"scored": count, "alerts": alert_count}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=PROCESSED_DIR / "features.parquet")
    parser.add_argument("--database", type=Path, default=SERVING_DB_PATH)
    parser.add_argument("--limit", type=int, default=300)
    args = parser.parse_args()
    result = seed_demo_alerts(FraudScorer(), ScoreRepository(args.database), args.features, args.limit)
    print(json.dumps({**result, "database": str(args.database)}, indent=2))


if __name__ == "__main__":
    main()
