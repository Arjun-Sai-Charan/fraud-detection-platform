"""Shared Feast-online and XGBoost scorer used by HTTP and Kafka paths."""
from __future__ import annotations

import json
import logging
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
from xgboost import DMatrix

from src.common.config import FEATURE_REPO_DIR, GNN_ARTIFACT_DIR, XGB_ARTIFACT_DIR
from src.common.model_registry import serving_model_version

_FEAST_DEPRECATION = (
    "_list_feature_views will make breaking changes. Please use _list_batch_feature_views instead. "
    "_list_feature_views will behave like _list_all_feature_views in the future."
)


class _FeastDeprecationFilter(logging.Filter):
    """Suppress only Feast 0.40's per-query API migration warning."""

    def filter(self, record: logging.LogRecord) -> bool:
        return record.getMessage() != _FEAST_DEPRECATION


def _install_feast_deprecation_filter() -> None:
    root = logging.getLogger()
    if not any(isinstance(item, _FeastDeprecationFilter) for item in root.filters):
        root.addFilter(_FeastDeprecationFilter())


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
        gnn_artifact_dir: Path = GNN_ARTIFACT_DIR,
    ) -> None:
        _install_feast_deprecation_filter()
        if model_source not in {"local", "registry"}:
            raise ValueError("model_source must be 'local' or 'registry'")
        self.artifact_dir = Path(artifact_dir)
        self.model_source = model_source
        self.tracking_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        self.model_name = model_name
        self.model_version = model_version
        self.feature_repo = Path(feature_repo)
        self.gnn_artifact_dir = Path(gnn_artifact_dir)
        self.model = model
        self.feature_store = feature_store
        self.feature_list = feature_list
        self.threshold = threshold
        self._injected_model = model is not None
        self._injected_store = feature_store is not None
        self.gnn_scores: dict[int, float] = {}
        self.rings: dict[int, tuple[str | None, int | None]] = {}
        self.blend_config: dict[str, Any] | None = None
        self._gnn_artifacts_loaded = False

    def _load_gnn_artifacts(self) -> None:
        if self._gnn_artifacts_loaded:
            return
        scores_path = self.gnn_artifact_dir / "gnn_scores.parquet"
        rings_path = self.gnn_artifact_dir / "rings.parquet"
        config_path = self.gnn_artifact_dir / "blend_config.json"
        if scores_path.is_file() and config_path.is_file():
            scores = pd.read_parquet(scores_path, columns=["transaction_id", "gnn_score"])
            self.gnn_scores = {
                int(row.transaction_id): float(row.gnn_score)
                for row in scores.itertuples(index=False)
            }
            self.blend_config = json.loads(config_path.read_text(encoding="utf-8"))
        if rings_path.is_file():
            rings = pd.read_parquet(rings_path, columns=["transaction_id", "ring_id", "ring_size"])
            self.rings = {
                int(row.transaction_id): (row.ring_id, None if pd.isna(row.ring_size) else int(row.ring_size))
                for row in rings.itertuples(index=False)
            }
        self._gnn_artifacts_loaded = True

    def _load_registry_artifacts(self) -> tuple[Any, list[str], dict[str, Any]]:
        mlflow.set_tracking_uri(self.tracking_uri)
        client = MlflowClient(tracking_uri=self.tracking_uri)
        if self.model_version:
            version = serving_model_version(client, self.model_name, self.model_version)
        else:
            version = serving_model_version(client, self.model_name)
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
        self._load_gnn_artifacts()

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
        gnn_score = self.gnn_scores.get(int(transaction_id))
        final_score = score
        cutoff = float(self.threshold["threshold"])
        tie_id = int(self.threshold.get("tie_break_transaction_id_max", -1))
        if gnn_score is not None and self.blend_config is not None:
            weight = float(self.blend_config["blend_weight_xgb"])
            final_score = weight * score + (1.0 - weight) * gnn_score
            cutoff = float(self.blend_config["threshold"])
            tie_id = int(self.blend_config.get("tie_break_transaction_id_max", -1))
        is_alert = final_score > cutoff or (final_score == cutoff and int(transaction_id) <= tie_id)
        ring_id, ring_size = self.rings.get(int(transaction_id), (None, None))
        latency_ms = (time.perf_counter() - started) * 1000.0
        return {
            "transaction_id": int(transaction_id),
            "xgb_score": score,
            "alert": bool(is_alert),
            "gnn_score": gnn_score,
            "final_score": float(final_score),
            "ring_id": ring_id,
            "ring_size": ring_size,
            "latency_ms": float(latency_ms),
        }

    def feature_contributions(self, transaction_id: int, limit: int = 5) -> list[dict[str, Any]]:
        """Return the largest signed XGBoost SHAP contributions for one online row."""
        self._ensure_ready()
        assert self.feature_list is not None and self.model is not None
        refs = [f"transaction_features:{name}" for name in self.feature_list]
        online = self.feature_store.get_online_features(
            features=refs,
            entity_rows=[{"transaction_id": int(transaction_id)}],
            full_feature_names=True,
        )
        values = self._online_feature_values(online, self.feature_list)
        row = pd.DataFrame([[values[name] for name in self.feature_list]], columns=self.feature_list)
        row = row.apply(pd.to_numeric, errors="coerce").astype(np.float64)
        contributions = self.model.get_booster().predict(
            DMatrix(row, feature_names=self.feature_list), pred_contribs=True
        )[0]
        ranked = sorted(
            zip(self.feature_list, contributions[:-1]),
            key=lambda item: abs(float(item[1])),
            reverse=True,
        )[: max(0, int(limit))]
        return [
            {"feature": name, "contribution": float(value)}
            for name, value in ranked
        ]


def scorer_from_environment() -> FraudScorer:
    return FraudScorer(
        model_source=os.getenv("FRAUD_MODEL_SOURCE", "local"),
        tracking_uri=os.getenv("MLFLOW_TRACKING_URI"),
        model_version=os.getenv("FRAUD_MODEL_VERSION"),
    )
