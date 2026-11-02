"""Shared Feast-online and XGBoost scorer used by HTTP and Kafka paths."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
from feast import FeatureStore
from mlflow.tracking import MlflowClient

from src.common.config import FEATURE_REPO_DIR, XGB_ARTIFACT_DIR


class FraudScorer:
    def __init__(
        self,
        artifact_dir: Path = XGB_ARTIFACT_DIR,
        *,
        model_source: str = "local",
        tracking_uri: str | None = None,
        model_name: str = "fraud-xgb",
        model_version: str | None = None,
        feature_store: Any | None = None,
        model: Any | None = None,
        feature_list: list[str] | None = None,
        threshold: dict[str, Any] | None = None,
        feature_repo: Path = FEATURE_REPO_DIR,
    ) -> None:
        if model_source not in {"local", "registry"}:
            raise ValueError("model_source must be 'local' or 'registry'")
        self.artifact_dir = Path(artifact_dir)
        self.model_source = model_source
        self.tracking_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        self.model_name = model_name
        self.model_version = model_version
        self.feature_repo = Path(feature_repo)
        self.model = model
        self.feature_store = feature_store
        self.feature_list = feature_list
        self.threshold = threshold
        self._injected_model = model is not None
        self._injected_store = feature_store is not None

    def _load_registry_artifacts(self) -> tuple[Any, list[str], dict[str, Any]]:
        mlflow.set_tracking_uri(self.tracking_uri)
        client = MlflowClient(tracking_uri=self.tracking_uri)
        if self.model_version:
            version = client.get_model_version(self.model_name, str(self.model_version))
        else:
            versions = client.search_model_versions(f"name='{self.model_name}'")
            if not versions:
                raise FileNotFoundError(f"No registered versions found for {self.model_name!r}")
            version = max(versions, key=lambda item: int(item.version))
        uri = f"models:/{self.model_name}/{version.version}"
        model = mlflow.xgboost.load_model(uri)
        threshold_path = Path(client.download_artifacts(version.run_id, "model/serving_threshold.json"))
        feature_path = Path(client.download_artifacts(version.run_id, "model/feature_list.json"))
        return (
            model,
            json.loads(feature_path.read_text(encoding="utf-8"))["features"],
            json.loads(threshold_path.read_text(encoding="utf-8")),
        )

    def _ensure_ready(self) -> None:
        if self.model is None or self.feature_list is None or self.threshold is None:
            if self.model_source == "registry":
                self.model, self.feature_list, self.threshold = self._load_registry_artifacts()
            else:
                model_dir = self.artifact_dir / "model"
                self.model = mlflow.xgboost.load_model(str(model_dir))
                threshold_path = self.artifact_dir / "threshold.json"
                feature_path = self.artifact_dir / "feature_list.json"
                self.threshold = json.loads(threshold_path.read_text(encoding="utf-8"))
                self.feature_list = json.loads(feature_path.read_text(encoding="utf-8"))["features"]
        if self.feature_store is None:
            self.feature_store = FeatureStore(repo_path=str(self.feature_repo))

    @staticmethod
    def _online_feature_values(response: Any, feature_names: list[str]) -> dict[str, Any]:
        values = response.to_dict()
        row: dict[str, Any] = {}
        for name in feature_names:
            candidates = (f"transaction_features__{name}", f"transaction_features:{name}", name)
            key = next((candidate for candidate in candidates if candidate in values), None)
            if key is None:
                raise KeyError(f"Feast online response omitted required feature {name!r}")
            column = values[key]
            if len(column) != 1:
                raise ValueError(f"Expected one online value for {name!r}; got {len(column)}")
            row[name] = column[0]
        if not any(value is not None for value in row.values()):
            raise KeyError("Feast returned no features for the requested transaction")
        return row

    def score(self, transaction_id: int) -> dict[str, Any]:
        self._ensure_ready()
        assert self.feature_list is not None and self.threshold is not None and self.model is not None
        started = time.perf_counter()
        refs = [f"transaction_features:{name}" for name in self.feature_list]
        online = self.feature_store.get_online_features(
            features=refs,
            entity_rows=[{"transaction_id": int(transaction_id)}],
            full_feature_names=True,
        )
        values = self._online_feature_values(online, self.feature_list)
        row = pd.DataFrame([[values[name] for name in self.feature_list]], columns=self.feature_list)
        row = row.apply(pd.to_numeric, errors="coerce").astype(np.float64)
        if hasattr(self.model, "predict_proba"):
            score = float(self.model.predict_proba(row)[0, 1])
        else:
            score = float(np.asarray(self.model.predict(row)).reshape(-1)[0])
        cutoff = float(self.threshold["threshold"])
        tie_id = int(self.threshold.get("tie_break_transaction_id_max", -1))
        is_alert = score > cutoff or (score == cutoff and int(transaction_id) <= tie_id)
        latency_ms = (time.perf_counter() - started) * 1000.0
        return {
            "transaction_id": int(transaction_id),
            "xgb_score": score,
            "alert": bool(is_alert),
            "gnn_score": None,
            "final_score": score,
            "ring_id": None,
            "ring_size": None,
            "latency_ms": float(latency_ms),
        }


def scorer_from_environment() -> FraudScorer:
    return FraudScorer(
        model_source=os.getenv("FRAUD_MODEL_SOURCE", "local"),
        tracking_uri=os.getenv("MLFLOW_TRACKING_URI"),
        model_version=os.getenv("FRAUD_MODEL_VERSION"),
    )
