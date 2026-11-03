"""Validation-gated, candidate-only XGBoost retraining trigger."""
from __future__ import annotations

import argparse
import json
import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from src.common.config import (
    DRIFT_SHARE_THRESHOLD,
    MIN_ANALYST_FEEDBACK,
    PROCESSED_DIR,
    RESULTS_DIR,
    SERVING_DB_PATH,
)
from src.models.train_xgb import train as train_xgb

DEFAULT_DRIFT_REPORT = RESULTS_DIR / "drift" / "test_data_drift.json"
DEFAULT_LOG_PATH = RESULTS_DIR / "retrain_log.json"
FEATURES_FILE = PROCESSED_DIR / "features.parquet"
MODEL_NAME = "fraud-xgb"
CONFIRMED_LABELS = {
    "fraud": 1,
    "confirmed_fraud": 1,
    "confirmed fraud": 1,
    "legit": 0,
    "legitimate": 0,
    "not_fraud": 0,
    "not fraud": 0,
    "benign": 0,
}
CONFIRMED_STATUSES = {"closed", "resolved", "confirmed"}


def should_trigger(
    drift_share: float | None,
    feedback_count: int,
    *,
    drift_threshold: float = DRIFT_SHARE_THRESHOLD,
    min_feedback: int = MIN_ANALYST_FEEDBACK,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if drift_share is not None and drift_share > drift_threshold:
        reasons.append(f"data_drift_share={drift_share:.6f}>{drift_threshold:.6f}")
    if feedback_count >= min_feedback:
        reasons.append(f"analyst_feedback_count={feedback_count}>={min_feedback}")
    return bool(reasons), reasons


def _feedback_count(db_path: Path) -> int:
    if not db_path.exists():
        return 0
    with sqlite3.connect(db_path) as connection:
        has_table = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='feedback'"
        ).fetchone()
        if not has_table:
            return 0
        return int(connection.execute("SELECT COUNT(*) FROM feedback").fetchone()[0])


def load_confirmed_feedback_rows(db_path: Path, features_path: Path) -> pd.DataFrame:
    """Map resolved analyst labels onto test replay features for candidate training."""
    if not db_path.exists():
        return pd.DataFrame()
    with sqlite3.connect(db_path) as connection:
        has_table = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='feedback'"
        ).fetchone()
        if not has_table:
            return pd.DataFrame()
        labels = pd.read_sql_query(
            "SELECT transaction_id, analyst_label, case_status FROM feedback ORDER BY id", connection
        )
    labels["normalized_label"] = labels["analyst_label"].astype(str).str.strip().str.lower()
    labels["normalized_status"] = labels["case_status"].astype(str).str.strip().str.lower()
    labels = labels.loc[
        labels["normalized_label"].isin(CONFIRMED_LABELS)
        & labels["normalized_status"].isin(CONFIRMED_STATUSES)
    ].drop_duplicates(subset="transaction_id", keep="last").copy()
    if labels.empty:
        return pd.DataFrame()
    features = pd.read_parquet(features_path)
    features = features.loc[features["split"] == "test"]
    merged = labels.merge(features, on="transaction_id", how="inner", suffixes=("_feedback", ""), validate="many_to_one")
    if merged.empty:
        return pd.DataFrame()
    merged["isFraud"] = merged["normalized_label"].map(CONFIRMED_LABELS).astype("int8")
    return merged


def _read_drift_share(path: Path) -> float | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    value = payload.get("drift_share")
    return None if value is None else float(value)


def run_retraining(
    *,
    dry_run: bool = False,
    include_feedback: bool = False,
    tracking_uri: str = "http://localhost:5000",
    drift_report: Path = DEFAULT_DRIFT_REPORT,
    feedback_db: Path = SERVING_DB_PATH,
    features_path: Path = FEATURES_FILE,
    output_path: Path = DEFAULT_LOG_PATH,
    drift_threshold: float = DRIFT_SHARE_THRESHOLD,
    min_feedback: int = MIN_ANALYST_FEEDBACK,
    train_callable: Callable[..., dict[str, Any]] = train_xgb,
) -> dict[str, Any]:
    drift_share = _read_drift_share(drift_report)
    feedback_count = _feedback_count(feedback_db)
    triggered, reasons = should_trigger(
        drift_share, feedback_count, drift_threshold=drift_threshold, min_feedback=min_feedback
    )
    entry: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "tracking_uri": tracking_uri,
        "drift_report": str(drift_report),
        "drift_share": drift_share,
        "drift_share_threshold": drift_threshold,
        "feedback_count": feedback_count,
        "feedback_threshold": min_feedback,
        "triggered": triggered,
        "trigger_reasons": reasons,
        "dry_run": bool(dry_run),
        "include_feedback_requested": bool(include_feedback),
        "model_name": MODEL_NAME,
        "registered_model_alias": None,
        "status": "dry_run" if dry_run else ("no_trigger" if not triggered else "pending"),
    }
    if not triggered or dry_run:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(entry, indent=2) + "\n", encoding="utf-8")
        return entry

    feedback_rows = load_confirmed_feedback_rows(feedback_db, features_path) if include_feedback else None
    if feedback_rows is not None and feedback_rows.empty:
        feedback_rows = None
    entry["confirmed_feedback_rows_used"] = 0 if feedback_rows is None else int(len(feedback_rows))
    reason_text = ";".join(reasons)
    try:
        with tempfile.TemporaryDirectory(prefix="fraud-retrain-") as temp_dir:
            metrics = train_callable(
                tracking_uri,
                features_path,
                trigger_reason=reason_text,
                feedback_rows=feedback_rows,
                output_metrics_path=None,
                pr_curve_path=Path(temp_dir) / "candidate_pr_curve.png",
                register_alias="candidate",
            )
        entry.update({
            "status": "candidate_registered",
            "registered_model_version": metrics.get("registered_model_version"),
            "registered_model_alias": "candidate",
            "selected_model": metrics.get("selected_model"),
            "validation_pr_auc": metrics["xgboost_runs"][metrics["selected_model"]]["splits"]["validation"]["pr_auc"],
            "evaluation_scope": metrics.get("evaluation_scope", "validation_and_test"),
            "confirmed_feedback_rows_used": metrics.get("feedback_training_rows", 0),
        })
    except Exception as exc:
        entry.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(entry, indent=2) + "\n", encoding="utf-8")
        raise
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(entry, indent=2) + "\n", encoding="utf-8")
    return entry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--include-feedback", action="store_true")
    parser.add_argument("--tracking-uri", default="http://localhost:5000")
    parser.add_argument("--drift-report", type=Path, default=DEFAULT_DRIFT_REPORT)
    parser.add_argument("--feedback-db", type=Path, default=SERVING_DB_PATH)
    parser.add_argument("--features", type=Path, default=FEATURES_FILE)
    parser.add_argument("--output", type=Path, default=DEFAULT_LOG_PATH)
    args = parser.parse_args()
    try:
        result = run_retraining(
            dry_run=args.dry_run,
            include_feedback=args.include_feedback,
            tracking_uri=args.tracking_uri,
            drift_report=args.drift_report,
            feedback_db=args.feedback_db,
            features_path=args.features,
            output_path=args.output,
        )
    except Exception as exc:
        parser.exit(1, f"Retraining failed: {type(exc).__name__}: {exc}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
