"""Train deterministic fraud baselines and XGBoost models with MLflow tracking."""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
from feast import FeatureStore
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from xgboost import XGBClassifier

from src.common.config import PROCESSED_DIR, RESULTS_DIR

SEED = 42
FEATURES_PATH = PROCESSED_DIR / "features.parquet"
FEATURE_REPO = Path(__file__).resolve().parents[1] / "feature_repo"
METRICS_PATH = RESULTS_DIR / "xgb_metrics.json"
PR_CURVE_PATH = RESULTS_DIR / "xgb_pr_curve.png"
EXCLUDED_COLUMNS = frozenset({"transaction_id", "event_timestamp", "uid", "isFraud", "split"})
ALERT_BUDGETS = (0.01, 0.02, 0.03, 0.05)
TRACKING_EXPERIMENT = "fraud_xgboost_phase3"
MODEL_NAME = "fraud-xgb"


def get_model_features(frame: pd.DataFrame) -> list[str]:
    """Choose numeric model inputs and enforce the requested exclusions."""
    excluded = EXCLUDED_COLUMNS.intersection(frame.columns)
    features = [
        column
        for column in frame.columns
        if column not in excluded and pd.api.types.is_numeric_dtype(frame[column].dtype)
    ]
    if not features:
        raise ValueError("No numeric model features remain after exclusions")
    if set(features).intersection(EXCLUDED_COLUMNS):
        raise AssertionError("Feature selection included an excluded ID or label column")
    return sorted(features)


def top_budget_mask(scores: np.ndarray, budget: float, transaction_ids: np.ndarray | None = None) -> tuple[np.ndarray, float]:
    """Select exactly ceil(n * budget) rows, ranking ties by transaction ID."""
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1 or len(scores) == 0:
        raise ValueError("scores must be a non-empty 1D array")
    if not 0 < budget <= 1:
        raise ValueError("budget must be in (0, 1]")
    if transaction_ids is None:
        transaction_ids = np.arange(len(scores), dtype=np.int64)
    transaction_ids = np.asarray(transaction_ids)
    if len(transaction_ids) != len(scores):
        raise ValueError("transaction_ids and scores must have equal length")
    alert_count = min(len(scores), max(1, math.ceil(len(scores) * budget)))
    order = np.lexsort((transaction_ids, -scores))
    selected = order[:alert_count]
    mask = np.zeros(len(scores), dtype=bool)
    mask[selected] = True
    return mask, float(scores[selected[-1]])


def fit_serving_threshold(
    validation_scores: np.ndarray,
    validation_ids: np.ndarray,
    budget: float = 0.03,
) -> dict[str, Any]:
    mask, threshold = top_budget_mask(validation_scores, budget, validation_ids)
    selected_ids = np.asarray(validation_ids)[mask]
    boundary_id = int(selected_ids[np.asarray(validation_scores)[mask] == threshold].max())
    return {
        "threshold": threshold,
        "alert_budget": budget,
        "validation_alerts": int(mask.sum()),
        "validation_rows": int(len(mask)),
        "tie_break": "transaction_id ascending among scores equal to threshold",
        "tie_break_transaction_id_max": boundary_id,
    }


def apply_serving_threshold(scores: np.ndarray, transaction_ids: np.ndarray, threshold: dict[str, Any]) -> np.ndarray:
    """Apply the validation score cutoff with its deterministic tie boundary."""
    scores = np.asarray(scores, dtype=np.float64)
    transaction_ids = np.asarray(transaction_ids)
    cutoff = float(threshold["threshold"])
    tie_id = int(threshold["tie_break_transaction_id_max"])
    return (scores > cutoff) | ((scores == cutoff) & (transaction_ids <= tie_id))


def _fetch_from_feast(features_path: Path) -> pd.DataFrame:
    source = pd.read_parquet(features_path, columns=["transaction_id", "event_timestamp"])
    entity_df = source.copy()
    entity_df["transaction_id"] = entity_df["transaction_id"].astype("int64")
    store = FeatureStore(repo_path=str(FEATURE_REPO))
    view = store.get_feature_view("transaction_features")
    refs = [
        f"transaction_features:{field.name}"
        for field in view.schema
        if field.name != "transaction_id"
    ]
    historical = store.get_historical_features(
        entity_df=entity_df,
        features=refs,
        full_feature_names=False,
    ).to_df()
    labels_and_metadata = pd.read_parquet(
        features_path,
        columns=["transaction_id", "transaction_dt", "isFraud", "split"],
    )
    result = historical.merge(labels_and_metadata, on="transaction_id", how="left", validate="one_to_one")
    if len(result) != len(source) or result["isFraud"].isna().any() or result["split"].isna().any():
        raise ValueError("Feast historical retrieval did not return complete training rows")
    return result


def load_training_data(features_path: Path = FEATURES_PATH, timeout_seconds: float = 300.0) -> tuple[pd.DataFrame, str, str | None, float]:
    """Try Feast historical retrieval, falling back to the prepared parquet file."""
    started = time.monotonic()
    try:
        result = _fetch_from_feast(features_path)
        elapsed = time.monotonic() - started
        if elapsed > timeout_seconds:
            raise TimeoutError(f"Feast historical retrieval took {elapsed:.1f}s (limit {timeout_seconds:.0f}s)")
        return result, "feast_historical", None, elapsed
    except Exception as exc:
        elapsed = time.monotonic() - started
        reason = f"{type(exc).__name__}: {exc}"
        fallback = pd.read_parquet(features_path)
        return fallback, "features_parquet_fallback", reason, elapsed


def _split_arrays(frame: pd.DataFrame, feature_names: list[str]) -> dict[str, tuple[pd.DataFrame, np.ndarray, np.ndarray]]:
    splits = {}
    for name in ("train", "validation", "test"):
        part = frame.loc[frame["split"] == name].sort_values("transaction_id", kind="mergesort")
        x = part.loc[:, feature_names].astype(np.float64)
        y = part["isFraud"].to_numpy(dtype=np.int8)
        ids = part["transaction_id"].to_numpy(dtype=np.int64)
        if not len(part) or y.min() == y.max():
            raise ValueError(f"Split {name} must contain both classes and at least one row")
        splits[name] = (x, y, ids)
    return splits


def _classification_metrics(y_true: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    return {
        "pr_auc": float(average_precision_score(y_true, scores)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "base_rate": float(np.mean(y_true)),
    }


def _budget_metrics(y_true: np.ndarray, scores: np.ndarray, ids: np.ndarray) -> dict[str, dict[str, float | int]]:
    result: dict[str, dict[str, float | int]] = {}
    positives = int(np.sum(y_true == 1))
    for budget in ALERT_BUDGETS:
        mask, threshold = top_budget_mask(scores, budget, ids)
        tp = int(np.sum(mask & (y_true == 1)))
        result[f"{int(budget * 100)}%"] = {
            "budget": budget,
            "alerts": int(mask.sum()),
            "alert_rate": float(mask.mean()),
            "threshold_score": threshold,
            "precision": float(tp / mask.sum()) if mask.any() else 0.0,
            "recall": float(tp / positives) if positives else 0.0,
        }
    return result


def _confusion_at_threshold(y_true: np.ndarray, scores: np.ndarray, ids: np.ndarray, threshold: dict[str, Any]) -> dict[str, int | float]:
    predicted = apply_serving_threshold(scores, ids, threshold)
    tp = int(np.sum(predicted & (y_true == 1)))
    fp = int(np.sum(predicted & (y_true == 0)))
    tn = int(np.sum(~predicted & (y_true == 0)))
    fn = int(np.sum(~predicted & (y_true == 1)))
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "alerts": tp + fp,
        "alert_rate": float((tp + fp) / len(y_true)),
        "precision": float(tp / (tp + fp)) if tp + fp else 0.0,
        "recall": float(tp / (tp + fn)) if tp + fn else 0.0,
    }


def _plot_pr_curves(
    predictions: dict[str, dict[str, np.ndarray]],
    labels: dict[str, np.ndarray],
    base_rates: dict[str, float],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))
    colors = {"unweighted": "#2878B5", "weighted": "#D95319"}
    styles = {"validation": "-", "test": "--"}
    for model_name, by_split in predictions.items():
        for split_name, scores in by_split.items():
            precision, recall, _ = precision_recall_curve(labels[split_name], scores)
            ap = average_precision_score(labels[split_name], scores)
            ax.plot(
                recall,
                precision,
                color=colors[model_name],
                linestyle=styles[split_name],
                label=f"{model_name} {split_name} (PR-AUC={ap:.4f})",
            )
    for split_name, rate in base_rates.items():
        ax.axhline(rate, color="#666666", linestyle=":" if split_name == "validation" else "-.",
                   label=f"{split_name} base rate={rate:.4%}")
    ax.set(xlabel="Recall", ylabel="Precision", title="XGBoost precision-recall curves by split")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.2)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _xgb_params(scale_pos_weight: float) -> dict[str, Any]:
    return {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "n_estimators": 1000,
        "early_stopping_rounds": 50,
        "max_depth": 6,
        "learning_rate": 0.05,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "max_bin": 256,
        "tree_method": "hist",
        "device": "cpu",
        "n_jobs": 1,
        "random_state": SEED,
        "scale_pos_weight": float(scale_pos_weight),
        "verbosity": 0,
    }


def _log_xgb_run(
    name: str,
    params: dict[str, Any],
    splits: dict[str, tuple[pd.DataFrame, np.ndarray, np.ndarray]],
    base_rates: dict[str, float],
    feature_names: list[str],
) -> tuple[dict[str, Any], dict[str, np.ndarray], str]:
    x_train, y_train, _ = splits["train"]
    x_val, y_val, val_ids = splits["validation"]
    xgb_model = XGBClassifier(**params)
    with mlflow.start_run(run_name=f"xgboost_{name}") as active_run:
        mlflow.set_tags({"model_family": "xgboost", "imbalance_variant": name})
        mlflow.log_params({
            **params,
            "seed": SEED,
            "train_rows": len(y_train),
            "feature_count": len(feature_names),
            "validation_base_rate": base_rates["validation"],
            "test_base_rate": base_rates["test"],
        })
        xgb_model.fit(
            x_train,
            y_train,
            eval_set=[(x_val, y_val)],
            verbose=False,
        )
        probabilities = {
            split_name: xgb_model.predict_proba(part_x)[:, 1]
            for split_name, (part_x, _, _) in splits.items()
        }
        metrics: dict[str, Any] = {"params": params, "splits": {}, "operating_points": {}}
        for split_name, (_, y_true, ids) in splits.items():
            measured = _classification_metrics(y_true, probabilities[split_name])
            metrics["splits"][split_name] = measured
            if split_name in ("validation", "test"):
                metrics["operating_points"][split_name] = _budget_metrics(y_true, probabilities[split_name], ids)
            mlflow.log_metrics({f"{split_name}_{key}": value for key, value in measured.items()})

        threshold = fit_serving_threshold(probabilities["validation"], val_ids, 0.03)
        metrics["serving_threshold"] = threshold
        metrics["validation_confusion_at_serving_threshold"] = _confusion_at_threshold(
            y_val, probabilities["validation"], val_ids, threshold
        )
        _, y_test, test_ids = splits["test"]
        metrics["test_confusion_at_serving_threshold"] = _confusion_at_threshold(
            y_test, probabilities["test"], test_ids, threshold
        )
        mlflow.log_metrics({
            "validation_precision_at_serving_threshold": metrics["validation_confusion_at_serving_threshold"]["precision"],
            "validation_recall_at_serving_threshold": metrics["validation_confusion_at_serving_threshold"]["recall"],
            "test_precision_at_serving_threshold": metrics["test_confusion_at_serving_threshold"]["precision"],
            "test_recall_at_serving_threshold": metrics["test_confusion_at_serving_threshold"]["recall"],
            "best_iteration": float(xgb_model.best_iteration),
        })

        importance = {
            feature: float(value)
            for feature, value in sorted(
                zip(feature_names, xgb_model.feature_importances_),
                key=lambda pair: (-float(pair[1]), pair[0]),
            )
        }
        metrics["feature_importance"] = importance
        mlflow.log_dict(importance, "feature_importance.json")
        mlflow.xgboost.log_model(
            xgb_model,
            artifact_path="model",
            input_example=x_train.head(3),
        )
        mlflow.log_dict(threshold, "model/serving_threshold.json")
        mlflow.log_dict({"features": feature_names}, "model/feature_list.json")
        run_id = active_run.info.run_id
    return metrics, probabilities, run_id


def _log_logistic_baseline(splits: dict[str, tuple[pd.DataFrame, np.ndarray, np.ndarray]], feature_names: list[str]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    x_train, y_train, _ = splits["train"]
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("logistic_regression", LogisticRegression(solver="lbfgs", max_iter=1000, random_state=SEED, n_jobs=1)),
    ])
    with mlflow.start_run(run_name="baseline_logistic_regression"):
        mlflow.set_tags({"model_family": "logistic_regression", "role": "baseline"})
        mlflow.log_params({"imputer": "median", "scaler": "StandardScaler", "solver": "lbfgs", "max_iter": 1000, "random_state": SEED, "feature_count": len(feature_names)})
        pipeline.fit(x_train, y_train)
        probabilities = {name: pipeline.predict_proba(x)[:, 1] for name, (x, _, _) in splits.items()}
        measured = {}
        for name, (_, y, _) in splits.items():
            measured[name] = _classification_metrics(y, probabilities[name])
            mlflow.log_metrics({f"{name}_{key}": value for key, value in measured[name].items()})
    return measured, probabilities


def train(tracking_uri: str = "http://localhost:5000", features_path: Path = FEATURES_PATH) -> dict[str, Any]:
    np.random.seed(SEED)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(TRACKING_EXPERIMENT)

    frame, feature_source, fallback_reason, retrieval_seconds = load_training_data(features_path)
    feature_names = get_model_features(frame)
    splits = _split_arrays(frame, feature_names)
    base_rates = {name: float(np.mean(y)) for name, (_, y, _) in splits.items()}
    train_negatives = int(np.sum(splits["train"][1] == 0))
    train_positives = int(np.sum(splits["train"][1] == 1))
    scale_pos_weight = train_negatives / train_positives

    results: dict[str, Any] = {
        "tracking_uri": tracking_uri,
        "experiment_name": TRACKING_EXPERIMENT,
        "seed": SEED,
        "feature_source": feature_source,
        "feature_retrieval_seconds": retrieval_seconds,
        "feature_fallback_reason": fallback_reason,
        "excluded_columns": sorted(EXCLUDED_COLUMNS),
        "feature_count": len(feature_names),
        "feature_list": feature_names,
        "split_rows": {name: int(len(y)) for name, (_, y, _) in splits.items()},
        "split_base_rates": base_rates,
        "random_baseline": {name: {"pr_auc": rate, "base_rate": rate} for name, rate in base_rates.items()},
        "logistic_regression_baseline": {},
        "xgboost_runs": {},
    }

    with threadpool_limits(limits=1):
        logistic_metrics, logistic_scores = _log_logistic_baseline(splits, feature_names)
        results["logistic_regression_baseline"] = {
            "metrics": logistic_metrics,
            "operating_points": {
                name: _budget_metrics(y, logistic_scores[name], ids)
                for name, (_, y, ids) in splits.items()
                if name in ("validation", "test")
            },
        }
        predictions: dict[str, dict[str, np.ndarray]] = {}
        run_ids = {}
        for name, weight in (("unweighted", 1.0), ("weighted", scale_pos_weight)):
            params = _xgb_params(weight)
            run_metrics, run_predictions, run_id = _log_xgb_run(name, params, splits, base_rates, feature_names)
            results["xgboost_runs"][name] = run_metrics
            predictions[name] = {split: run_predictions[split] for split in ("validation", "test")}
            run_ids[name] = run_id

    best_name = max(results["xgboost_runs"], key=lambda name: results["xgboost_runs"][name]["splits"]["validation"]["pr_auc"])
    best_model_run_id = run_ids[best_name]
    best_run = results["xgboost_runs"][best_name]
    _, y_val, val_ids = splits["validation"]
    _, y_test, test_ids = splits["test"]
    serving_threshold = best_run["serving_threshold"]
    results["selected_model"] = best_name
    results["registered_model_name"] = MODEL_NAME
    results["serving_threshold"] = serving_threshold
    results["validation_confusion_at_serving_threshold"] = best_run["validation_confusion_at_serving_threshold"]
    results["test_confusion_at_serving_threshold"] = best_run["test_confusion_at_serving_threshold"]

    _plot_pr_curves(
        predictions,
        {name: splits[name][1] for name in ("validation", "test")},
        {name: base_rates[name] for name in ("validation", "test")},
        PR_CURVE_PATH,
    )
    with mlflow.start_run(run_id=best_model_run_id):
        mlflow.set_tag("registered_best_model", "true")
        mlflow.log_artifact(str(PR_CURVE_PATH), artifact_path="evaluation")
        mlflow.log_dict(results, "phase3_metrics.json")
    for run_id in run_ids.values():
        with mlflow.start_run(run_id=run_id):
            mlflow.log_artifact(str(PR_CURVE_PATH), artifact_path="evaluation")

    model_uri = f"runs:/{best_model_run_id}/model"
    registered = mlflow.register_model(model_uri, MODEL_NAME)
    results["registered_model_version"] = int(registered.version)
    results["registered_model_source"] = model_uri
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracking-uri", default="http://localhost:5000")
    parser.add_argument("--features", type=Path, default=FEATURES_PATH)
    args = parser.parse_args()
    try:
        result = train(args.tracking_uri, args.features)
    except Exception as exc:
        parser.exit(1, f"Training failed: {type(exc).__name__}: {exc}\n")
    print(json.dumps({
        "tracking_uri": result["tracking_uri"],
        "feature_source": result["feature_source"],
        "feature_count": result["feature_count"],
        "split_base_rates": result["split_base_rates"],
        "selected_model": result["selected_model"],
        "registered_model_version": result["registered_model_version"],
        "serving_threshold": result["serving_threshold"],
        "validation_confusion_at_serving_threshold": result["validation_confusion_at_serving_threshold"],
        "test_confusion_at_serving_threshold": result["test_confusion_at_serving_threshold"],
    }, indent=2))


if __name__ == "__main__":
    main()
