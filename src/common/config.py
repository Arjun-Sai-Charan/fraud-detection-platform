"""Shared project paths and constants."""
from pathlib import Path
import os

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

