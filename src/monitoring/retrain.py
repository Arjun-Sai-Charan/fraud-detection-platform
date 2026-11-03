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
import mlflow
import mlflow.xgboost
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from threadpoolctl import threadpool_limits
from xgboost import XGBClassifier
from mlflow.exceptions import MlflowException

from src.common.config import (
    DRIFT_SHARE_THRESHOLD,
    MIN_ANALYST_FEEDBACK,
    PROCESSED_DIR,
    RESULTS_DIR,
    SERVING_DB_PATH,
)
from src.models.train_xgb import train as train_xgb
from src.models.train_xgb import _budget_metrics, _split_arrays, get_model_features, load_training_data
from src.common.model_registry import serving_model_version
from mlflow.tracking import MlflowClient

DEFAULT_DRIFT_REPORT = RESULTS_DIR / "drift" / "recent_window_drift.json"
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
EXPANDED_TRAIN_ROWS = 120_000
ORIGINAL_TRAIN_ROWS = 90_000


def expanded_window_sizes(train_rows: int, validation_rows: int, test_rows: int) -> dict[str, int]:
    return {
        "original_train_rows": int(train_rows),
        "validation_rows_added": int(validation_rows),
        "expanded_train_rows": int(train_rows + validation_rows),
        "test_rows": int(test_rows),
    }


def expanded_n_estimators(original_best_iteration: int, expanded_rows: int, original_rows: int) -> int:
    if original_best_iteration <= 0 or original_rows <= 0 or expanded_rows <= 0:
        raise ValueError("best iteration and training row counts must be positive")
    return int(round(original_best_iteration * expanded_rows / original_rows))


def _typed_original_params(run_params: dict[str, str], n_estimators: int) -> dict[str, Any]:
    integer_params = {"max_depth", "max_bin", "n_jobs", "random_state", "verbosity"}
    float_params = {"learning_rate", "min_child_weight", "subsample", "colsample_bytree", "reg_lambda", "scale_pos_weight"}
    string_params = {"objective", "eval_metric", "tree_method", "device"}
    params: dict[str, Any] = {}
    for name in integer_params | float_params | string_params:
        if name not in run_params:
            continue
        if name in integer_params:
            params[name] = int(float(run_params[name]))
        elif name in float_params:
            params[name] = float(run_params[name])
        else:
            params[name] = run_params[name]
    params["n_estimators"] = int(n_estimators)
    # Intentionally omit early_stopping_rounds: validation is part of training.
    return params


def train_expanded_candidate(
    tracking_uri: str,
    features_path: Path,
    trigger_reason: str,
) -> dict[str, Any]:
    """Fit champion hyperparameters on train+validation and compare once on test."""
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("fraud_xgboost_phase3")
    client = MlflowClient(tracking_uri=tracking_uri)
    champion_version = serving_model_version(client, MODEL_NAME)
    champion_run = client.get_run(champion_version.run_id)
    champion = mlflow.xgboost.load_model(f"models:/{MODEL_NAME}/{champion_version.version}")
    try:
        original_best_iteration = int(round(champion_run.data.metrics["best_iteration"]))
    except KeyError as exc:
        raise ValueError("The serving champion run has no logged best_iteration") from exc

    with tempfile.TemporaryDirectory(prefix="fraud-expanded-candidate-") as temp_dir:
        feature_path = Path(client.download_artifacts(
            champion_version.run_id, "model/feature_list.json", dst_path=temp_dir
        ))
        threshold_path = Path(client.download_artifacts(
            champion_version.run_id, "model/serving_threshold.json", dst_path=temp_dir
        ))
        feature_payload = json.loads(feature_path.read_text(encoding="utf-8"))
        threshold_payload = json.loads(threshold_path.read_text(encoding="utf-8"))

    frame, feature_source, fallback_reason, retrieval_seconds = load_training_data(features_path)
    feature_names = list(feature_payload["features"])
    model_features = get_model_features(frame)
    if model_features != sorted(feature_names):
        raise ValueError("Current features do not match the champion model's saved feature list")
    splits = _split_arrays(frame, feature_names)
    train_x, train_y, _ = splits["train"]
    validation_x, validation_y, _ = splits["validation"]
    test_x, test_y, test_ids = splits["test"]
    expanded_x = pd.concat([train_x, validation_x], ignore_index=True)
    expanded_y = np.concatenate([train_y, validation_y])
    window_sizes = expanded_window_sizes(len(train_y), len(validation_y), len(test_y))
    n_estimators = expanded_n_estimators(
        original_best_iteration,
        window_sizes["expanded_train_rows"],
        window_sizes["original_train_rows"],
    )
    params = _typed_original_params(champion_run.data.params, n_estimators)
    candidate = XGBClassifier(**params)
    with threadpool_limits(limits=1):
        candidate.fit(expanded_x, expanded_y, verbose=False)
        candidate_scores = candidate.predict_proba(test_x)[:, 1]
        champion_scores = champion.predict_proba(test_x)[:, 1]

    test_base_rate = float(np.mean(test_y))

    def test_metrics(scores: np.ndarray) -> dict[str, Any]:
        return {
            "pr_auc": float(average_precision_score(test_y, scores)),
            "roc_auc": float(roc_auc_score(test_y, scores)),
            "base_rate": test_base_rate,
            "alert_budgets": _budget_metrics(test_y, scores, test_ids),
        }

    champion_metrics = test_metrics(champion_scores)
    candidate_metrics = test_metrics(candidate_scores)
    comparison = {
        "champion_version": int(champion_version.version),
        "candidate_training_window": "train_plus_validation",
        "candidate_test_metrics": candidate_metrics,
        "champion_test_metrics": champion_metrics,
        "test_rows": int(len(test_y)),
        "test_used_for": "final champion-versus-candidate comparison only",
    }
    with mlflow.start_run(run_name="xgboost_retrain_expanded_candidate") as run:
        mlflow.set_tags({
            "model_family": "xgboost",
            "retraining_trigger_reason": trigger_reason,
            "training_window": "train_plus_validation",
            "candidate_role": "candidate",
            "test_usage": "final_champion_comparison_only",
        })
        mlflow.log_params({
            **params,
            "original_best_iteration": original_best_iteration,
            "original_train_rows": window_sizes["original_train_rows"],
            "validation_rows_added": window_sizes["validation_rows_added"],
            "expanded_train_rows": window_sizes["expanded_train_rows"],
            "test_rows": window_sizes["test_rows"],
            "early_stopping": False,
            "feature_source": feature_source,
        })
        mlflow.log_metrics({
            "candidate_test_pr_auc": candidate_metrics["pr_auc"],
            "candidate_test_roc_auc": candidate_metrics["roc_auc"],
            "champion_test_pr_auc": champion_metrics["pr_auc"],
            "champion_test_roc_auc": champion_metrics["roc_auc"],
            "candidate_test_base_rate": test_base_rate,
        })
        mlflow.log_dict(comparison, "candidate_vs_champion_test.json")
        mlflow.xgboost.log_model(candidate, artifact_path="model", input_example=expanded_x.head(3))
        mlflow.log_dict(threshold_payload, "model/serving_threshold.json")
        mlflow.log_dict(feature_payload, "model/feature_list.json")
        run_id = run.info.run_id

    # Preserve the former candidate's non-serving status when moving the alias.
    try:
        previous_candidate = client.get_model_version_by_alias(MODEL_NAME, "candidate")
    except MlflowException:
        previous_candidate = None
    if previous_candidate is not None:
        client.set_model_version_tag(
            MODEL_NAME, str(previous_candidate.version), "retraining_role", "candidate"
        )
    registered = mlflow.register_model(f"runs:/{run_id}/model", MODEL_NAME)
    client.set_model_version_tag(MODEL_NAME, str(registered.version), "retraining_role", "candidate")
    client.set_model_version_tag(MODEL_NAME, str(registered.version), "retraining_trigger_reason", trigger_reason)
    client.set_registered_model_alias(MODEL_NAME, "candidate", str(registered.version))
    return {
        "registered_model_version": int(registered.version),
        "registered_model_alias": "candidate",
        "registered_model_source": f"runs:/{run_id}/model",
        "champion_version": int(champion_version.version),
        "original_best_iteration": original_best_iteration,
        "n_estimators": n_estimators,
        "n_estimators_formula": f"round({original_best_iteration} * {window_sizes['expanded_train_rows']} / {window_sizes['original_train_rows']})",
        "window_sizes": window_sizes,
        "feature_source": feature_source,
        "feature_fallback_reason": fallback_reason,
        "feature_retrieval_seconds": retrieval_seconds,
        "candidate_test_metrics": candidate_metrics,
        "champion_test_metrics": champion_metrics,
        "test_used_for": comparison["test_used_for"],
    }


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
    window: str = "expanded",
    tracking_uri: str = "http://localhost:5000",
    drift_report: Path = DEFAULT_DRIFT_REPORT,
    feedback_db: Path = SERVING_DB_PATH,
    features_path: Path = FEATURES_FILE,
    output_path: Path = DEFAULT_LOG_PATH,
    drift_threshold: float = DRIFT_SHARE_THRESHOLD,
    min_feedback: int = MIN_ANALYST_FEEDBACK,
    train_callable: Callable[..., dict[str, Any]] = train_xgb,
) -> dict[str, Any]:
    if window not in {"expanded", "legacy"}:
        raise ValueError("window must be 'expanded' or 'legacy'")
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
        "window_requested": window,
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
        if feedback_rows is None and window == "expanded":
            metrics = train_expanded_candidate(tracking_uri, features_path, reason_text)
            entry.update({
                "status": "candidate_registered",
                "effective_window": "train_plus_validation",
                "registered_model_version": metrics["registered_model_version"],
                "registered_model_alias": "candidate",
                "champion_version": metrics["champion_version"],
                "expanded_training_rows": metrics["window_sizes"]["expanded_train_rows"],
                "original_best_iteration": metrics["original_best_iteration"],
                "n_estimators": metrics["n_estimators"],
                "n_estimators_formula": metrics["n_estimators_formula"],
                "feature_source": metrics["feature_source"],
                "feature_fallback_reason": metrics["feature_fallback_reason"],
                "candidate_vs_champion_test": {
                    "candidate": metrics["candidate_test_metrics"],
                    "champion": metrics["champion_test_metrics"],
                    "test_used_for": metrics["test_used_for"],
                },
                "evaluation_scope": "test_final_comparison_only",
            })
        else:
            # Preserve the Phase 6 analyst-feedback training/evaluation path.
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
                "effective_window": "feedback_train_validation" if feedback_rows is not None else "legacy",
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
    parser.add_argument("--window", choices=("expanded", "legacy"), default="expanded")
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
            window=args.window,
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
