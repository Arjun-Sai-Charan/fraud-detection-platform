"""Shared project paths and constants."""
from pathlib import Path
import os
import json

ROOT = Path(__file__).resolve().parents[2]
KAFKA_BOOTSTRAP_HOST = os.getenv("KAFKA_BOOTSTRAP_HOST", "localhost:9092")
KAFKA_BOOTSTRAP_DOCKER = os.getenv("KAFKA_BOOTSTRAP_DOCKER", "kafka:29092")
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = ROOT / "results"
DOCS_DIR = ROOT / "docs"
FEATURE_REPO_DIR = ROOT / "src" / "feature_repo"
XGB_ARTIFACT_DIR = ROOT / "artifacts" / "xgb"
GNN_ARTIFACT_DIR = ROOT / "artifacts" / "gnn"
SERVING_DB_PATH = DATA_DIR / "serving.db"
KAFKA_TRANSACTIONS_TOPIC = os.getenv("KAFKA_TRANSACTIONS_TOPIC", "transactions")
KAFKA_FRAUD_SCORES_TOPIC = os.getenv("KAFKA_FRAUD_SCORES_TOPIC", "fraud-scores")
TRANSACTION_FILE = RAW_DIR / "train_transaction.csv"
IDENTITY_FILE = RAW_DIR / "train_identity.csv"
SUBSAMPLE_ROWS = 150_000
TRAIN_FRACTION = 0.60
VALIDATION_FRACTION = 0.20
RANDOM_SEED = 42
ALERT_THRESHOLD = 0.5  # Replaced by a validation-selected value during model training.
DRIFT_NATURAL_BASELINE_PATH = RESULTS_DIR / "drift" / "natural_baseline.json"
DRIFT_THRESHOLD_MARGIN = 0.10


def _load_drift_share_threshold() -> float:
    configured = os.getenv("FRAUD_DRIFT_SHARE_THRESHOLD")
    if configured is not None:
        return float(configured)
    if DRIFT_NATURAL_BASELINE_PATH.is_file():
        baseline = json.loads(DRIFT_NATURAL_BASELINE_PATH.read_text(encoding="utf-8"))
        if "trigger_threshold" in baseline:
            return float(baseline["trigger_threshold"])
        return min(1.0, float(baseline["max"]) + DRIFT_THRESHOLD_MARGIN)
    return 0.30


DRIFT_SHARE_THRESHOLD = _load_drift_share_threshold()
MIN_ANALYST_FEEDBACK = int(os.getenv("FRAUD_MIN_ANALYST_FEEDBACK", "25"))

