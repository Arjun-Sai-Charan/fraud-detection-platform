"""Shared project paths and constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = ROOT / "results"
DOCS_DIR = ROOT / "docs"
TRANSACTION_FILE = RAW_DIR / "train_transaction.csv"
IDENTITY_FILE = RAW_DIR / "train_identity.csv"
SUBSAMPLE_ROWS = 150_000
TRAIN_FRACTION = 0.60
VALIDATION_FRACTION = 0.20
RANDOM_SEED = 42
ALERT_THRESHOLD = 0.5  # Replaced by a validation-selected value during model training.

