"""One persisted categorical encoding shared by offline and serving paths."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

from src.common.config import PROCESSED_DIR

MISSING_CATEGORY = "__MISSING__"
CATEGORICAL_COLUMNS = (
    "ProductCD",
    "card4",
    "card6",
    "P_emaildomain",
    "R_emaildomain",
    *(f"M{i}" for i in range(1, 10)),
)
DEFAULT_ENCODER_PATH = PROCESSED_DIR / "categorical_encoder.joblib"


def normalize_categories(frame: pd.DataFrame, columns: tuple[str, ...] = CATEGORICAL_COLUMNS) -> pd.DataFrame:
    """Normalize nulls and scalar values to the same string representation."""
    normalized = pd.DataFrame(index=frame.index)
    for column in columns:
        if column not in frame:
            raise ValueError(f"Missing categorical feature column: {column}")
        normalized[column] = frame[column].astype("string").fillna(MISSING_CATEGORY).astype(str)
    return normalized


def fit_encoder(
    training_frame: pd.DataFrame,
    artifact_path: Path = DEFAULT_ENCODER_PATH,
    columns: tuple[str, ...] = CATEGORICAL_COLUMNS,
) -> dict[str, Any]:
    """Fit once on training rows and persist the reusable encoder artifact."""
    normalized = normalize_categories(training_frame, columns)
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, dtype=np.float32)
    encoder.fit(normalized.loc[:, list(columns)])
    artifact = {"encoder": encoder, "columns": columns, "missing_category": MISSING_CATEGORY}
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, artifact_path)
    return artifact


def load_encoder(artifact_path: Path = DEFAULT_ENCODER_PATH) -> dict[str, Any]:
    """Load the saved encoding artifact for inference/serving."""
    return joblib.load(artifact_path)


def transform_categories(frame: pd.DataFrame, artifact: dict[str, Any]) -> pd.DataFrame:
    """Return normalized raw categoricals and encoded numeric columns."""
    columns = tuple(artifact["columns"])
    normalized = normalize_categories(frame, columns)
    result = normalized.copy()
    encoded = artifact["encoder"].transform(normalized.loc[:, list(columns)])
    for offset, column in enumerate(columns):
        result[f"{column}_encoded"] = encoded[:, offset]
    return result


def encode_serving_row(row: dict[str, Any], artifact_path: Path = DEFAULT_ENCODER_PATH) -> dict[str, Any]:
    """Encode one raw inference row with the fitted artifact, without refitting."""
    artifact = load_encoder(artifact_path)
    transformed = transform_categories(pd.DataFrame([row]), artifact)
    return transformed.iloc[0].to_dict()
