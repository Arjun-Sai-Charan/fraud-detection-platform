"""Generate Evidently 0.5.1 data and score drift reports."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import mlflow.xgboost
import numpy as np
import pandas as pd
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report

from src.common.config import (
    DRIFT_NATURAL_BASELINE_PATH,
    DRIFT_SHARE_THRESHOLD,
    DRIFT_THRESHOLD_MARGIN,
    PROCESSED_DIR,
    RESULTS_DIR,
    XGB_ARTIFACT_DIR,
)

FEATURES_PATH = PROCESSED_DIR / "features.parquet"
FEATURE_LIST_PATH = XGB_ARTIFACT_DIR / "feature_list.json"
DRIFT_DIR = RESULTS_DIR / "drift"
METRICS_PATH = RESULTS_DIR / "xgb_metrics.json"


def _feature_names() -> list[str]:
    if FEATURE_LIST_PATH.exists():
        payload = json.loads(FEATURE_LIST_PATH.read_text(encoding="utf-8"))
        return list(payload.get("features", payload))
    return list(json.loads(METRICS_PATH.read_text(encoding="utf-8"))["feature_list"])


def _evidently_summary(report: Report, reference_rows: int, current_rows: int) -> dict[str, Any]:
    """Extract stable fields from the installed Evidently 0.5.1 report API."""
    report_data = report.as_dict()
    dataset_result: dict[str, Any] = {}
    drift_table: dict[str, Any] = {}
    for metric in report_data["metrics"]:
        name = metric["metric"]
        result = metric["result"]
        if name == "DatasetDriftMetric":
            dataset_result = result
        elif name == "DataDriftTable":
            drift_table = result
    if not dataset_result:
        raise RuntimeError("Evidently 0.5.1 report did not return DatasetDriftMetric")
    by_column = drift_table.get("drift_by_columns", {})
    drifted = sorted(
        name for name, values in by_column.items() if bool(values.get("drift_detected"))
    )
    details = [
        {
            "name": name,
            "tags": ["cumulative_by_construction"] if is_cumulative_feature(name) else [],
        }
        for name in drifted
    ]
    tests = sorted({
        str(values["stattest_name"])
        for values in by_column.values()
        if values.get("stattest_name") is not None
    })
    return {
        "drift_share": float(dataset_result["share_of_drifted_columns"]),
        "evidently_drift_share_threshold": float(dataset_result.get("drift_share", 0.5)),
        "dataset_drift": bool(dataset_result["dataset_drift"]),
        "number_of_columns": int(dataset_result["number_of_columns"]),
        "number_of_drifted_columns": int(dataset_result["number_of_drifted_columns"]),
        "drifted_columns": drifted,
        "drifted_column_details": details,
        "test_names": tests,
        "column_tests": {
            name: {
                "stattest_name": values.get("stattest_name"),
                "drift_score": values.get("drift_score"),
                "drift_detected": bool(values.get("drift_detected")),
            }
            for name, values in sorted(by_column.items())
        },
        "reference_rows": int(reference_rows),
        "current_rows": int(current_rows),
    }


def is_cumulative_feature(name: str) -> bool:
    """Mark features whose values can grow/change as history accumulates."""
    normalized = name.lower()
    return (
        name.startswith("D")
        or "count" in normalized
        or name in {"uid_is_hub"}
    )


def _run_evidently(reference: pd.DataFrame, current: pd.DataFrame, columns: list[str]) -> Report:
    report = Report(metrics=[DataDriftPreset(columns=columns)])
    report.run(reference_data=reference.loc[:, columns], current_data=current.loc[:, columns])
    return report


def make_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    *,
    columns: list[str],
    report_name: str,
    output_dir: Path = DRIFT_DIR,
    drift_share_threshold: float = DRIFT_SHARE_THRESHOLD,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    missing = sorted(set(columns) - set(reference.columns) | (set(columns) - set(current.columns)))
    if missing:
        raise ValueError(f"Drift columns missing from input frames: {missing}")
    report = _run_evidently(reference, current, columns)
    summary = _evidently_summary(report, len(reference), len(current))
    summary.update({"report_name": report_name, "trigger_threshold": float(drift_share_threshold)})
    if metadata:
        summary.update(metadata)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{report_name}.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    report.save_html(str(output_dir / f"{report_name}.html"))
    return summary


def build_shifted_test(
    test: pd.DataFrame,
    feature_names: list[str],
    *,
    stress_multiplier: float = 20.0,
) -> pd.DataFrame:
    """Create a deliberately broad synthetic stress-test distribution."""
    shifted = test.copy()
    specifically_shifted: set[str] = set()
    for column, multiplier in (
        ("TransactionAmt", 2.5),
        ("txn_count_last_1h", 2.0),
        ("txn_count_last_24h", 2.0),
    ):
        if column in feature_names:
            shifted[column] = pd.to_numeric(shifted[column], errors="coerce") * multiplier
            specifically_shifted.add(column)
    if "device_change_vs_previous" in feature_names:
        shifted["device_change_vs_previous"] = 1.0
        specifically_shifted.add("device_change_vs_previous")
    # A handful of changed columns cannot clear a threshold calibrated across
    # all 52 features. Move the remaining columns as well, explicitly making
    # this a broad synthetic stress test rather than a plausible future batch.
    for column in feature_names:
        if column in specifically_shifted:
            continue
        values = pd.to_numeric(shifted[column], errors="coerce")
        scale = float(values.std(skipna=True))
        center = float(values.mean(skipna=True))
        if not np.isfinite(scale):
            scale = 0.0
        if not np.isfinite(center):
            center = 0.0
        offset = max(abs(scale), abs(center), 1.0) * stress_multiplier
        shifted[column] = values + offset
    return shifted


def natural_baseline_from_windows(
    windows: list[pd.DataFrame],
    feature_names: list[str],
) -> dict[str, Any]:
    if len(windows) < 2:
        raise ValueError("At least two consecutive windows are required")
    comparisons = []
    shares = []
    for index in range(1, len(windows)):
        reference = windows[index - 1].loc[:, feature_names].apply(pd.to_numeric, errors="coerce")
        current = windows[index].loc[:, feature_names].apply(pd.to_numeric, errors="coerce")
        summary = _evidently_summary(_run_evidently(reference, current, feature_names), len(reference), len(current))
        summary.update({"reference_window": index - 1, "current_window": index})
        comparisons.append(summary)
        shares.append(summary["drift_share"])
    values = np.asarray(shares, dtype=np.float64)
    maximum = float(np.max(values))
    threshold = natural_trigger_threshold(maximum)
    return {
        "window_size_rows": int(len(windows[0])),
        "window_count": len(windows),
        "comparison_count": len(comparisons),
        "drift_shares": [float(value) for value in values],
        "mean": float(np.mean(values)),
        "max": maximum,
        "p95": float(np.percentile(values, 95)),
        "threshold_rule": "natural baseline maximum + 0.10, capped at 1.0",
        "threshold_margin": DRIFT_THRESHOLD_MARGIN,
        "trigger_threshold": threshold,
        "comparisons": comparisons,
    }


def natural_trigger_threshold(natural_max: float, margin: float = DRIFT_THRESHOLD_MARGIN) -> float:
    if not 0.0 <= natural_max <= 1.0 or margin < 0:
        raise ValueError("natural drift share must be in [0,1] and margin must be non-negative")
    return min(1.0, float(natural_max) + float(margin))


def generate_reports(
    features_path: Path = FEATURES_PATH,
    output_dir: Path = DRIFT_DIR,
    drift_share_threshold: float | None = None,
    window_size: int = 15_000,
) -> dict[str, Any]:
    frame = pd.read_parquet(features_path)
    feature_names = _feature_names()
    if len(feature_names) != len(set(feature_names)):
        raise ValueError("Saved XGBoost feature list contains duplicate names")
    missing = sorted(set(feature_names) - set(frame.columns))
    if missing:
        raise ValueError(f"Saved feature columns missing from features parquet: {missing}")
    frame = frame.sort_values(["event_timestamp", "transaction_id"], kind="mergesort").reset_index(drop=True)
    split_frames = {
        name: frame.loc[frame["split"] == name].sort_values("transaction_id", kind="mergesort").copy()
        for name in ("train", "validation", "test")
    }
    if any(part.empty for part in split_frames.values()):
        raise ValueError("Expected non-empty train, validation, and test windows")
    windows = [frame.iloc[start:start + window_size].copy() for start in range(0, len(frame), window_size)]
    if any(len(window) != window_size for window in windows):
        raise ValueError("Timeline row count must be divisible by the configured natural window size")
    baseline = natural_baseline_from_windows(windows, feature_names)
    baseline["window_boundaries"] = [
        {
            "window": index,
            "start_transaction_id": int(window.iloc[0]["transaction_id"]),
            "end_transaction_id": int(window.iloc[-1]["transaction_id"]),
            "start_event_timestamp": str(window.iloc[0]["event_timestamp"]),
            "end_event_timestamp": str(window.iloc[-1]["event_timestamp"]),
        }
        for index, window in enumerate(windows)
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "natural_baseline.json").write_text(
        json.dumps(baseline, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    threshold = baseline["trigger_threshold"] if drift_share_threshold is None else float(drift_share_threshold)

    reference = split_frames["train"].loc[:, feature_names].apply(pd.to_numeric, errors="coerce")
    validation = split_frames["validation"].loc[:, feature_names].apply(pd.to_numeric, errors="coerce")
    test = split_frames["test"].loc[:, feature_names].apply(pd.to_numeric, errors="coerce")
    rates = {name: float(part["isFraud"].mean()) for name, part in split_frames.items()}
    results: dict[str, Any] = {
        "reference_split": "train",
        "feature_count": len(feature_names),
        "split_fraud_rates": rates,
        "natural_baseline": baseline,
        "drift_share_threshold": threshold,
    }
    for name, current in (("validation_data_drift", validation), ("test_data_drift", test)):
        results[name] = make_report(
            reference, current, columns=feature_names, report_name=name, output_dir=output_dir,
            drift_share_threshold=threshold,
            metadata={"reference_split": "train", "current_split": name.removesuffix("_data_drift"),
                      "reference_fraud_rate": rates["train"],
                      "current_fraud_rate": rates[name.removesuffix("_data_drift")],
                      "trigger_eligible": False,
                      "informational_only_reason": "vs-TRAIN comparison spans multiple natural windows"},
        )
    recent_reference = windows[-2].loc[:, feature_names].apply(pd.to_numeric, errors="coerce")
    recent_current = windows[-1].loc[:, feature_names].apply(pd.to_numeric, errors="coerce")
    results["recent_window_drift"] = make_report(
        recent_reference, recent_current, columns=feature_names, report_name="recent_window_drift",
        output_dir=output_dir, drift_share_threshold=threshold,
        metadata={"reference_window": len(windows) - 2, "current_window": len(windows) - 1,
                  "trigger_eligible": True, "synthetic_stress_test": False},
    )
    shifted = build_shifted_test(windows[-1], feature_names)
    results["shifted_window_drift"] = make_report(
        recent_reference, shifted, columns=feature_names, report_name="shifted_window_drift",
        output_dir=output_dir, drift_share_threshold=threshold,
        metadata={"reference_window": len(windows) - 2, "current_window": len(windows) - 1,
                  "trigger_eligible": False, "synthetic_stress_test": True,
                  "stress_multiplier": 20.0,
                  "shifts": {"TransactionAmt": 2.5, "txn_count_last_1h": 2.0,
                             "txn_count_last_24h": 2.0, "device_change_vs_previous": "set to 1",
                             "remaining_model_features": "shifted by 20 standard deviations or 20x magnitude"}},
    )

    model = mlflow.xgboost.load_model(str(XGB_ARTIFACT_DIR / "model"))
    validation_scores = model.predict_proba(validation)[:, 1]
    test_scores = model.predict_proba(test)[:, 1]
    results["score_drift"] = make_report(
        pd.DataFrame({"xgb_score": validation_scores}),
        pd.DataFrame({"xgb_score": test_scores}), columns=["xgb_score"], report_name="score_drift",
        output_dir=output_dir, drift_share_threshold=threshold,
        metadata={"reference_split": "validation", "current_split": "test",
                  "reference_fraud_rate": rates["validation"], "current_fraud_rate": rates["test"],
                  "trigger_eligible": False},
    )
    summary_path = output_dir / "drift_summary.json"
    summary_path.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=FEATURES_PATH)
    parser.add_argument("--output-dir", type=Path, default=DRIFT_DIR)
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args()
    try:
        result = generate_reports(args.features, args.output_dir, args.threshold)
    except Exception as exc:
        parser.exit(1, f"Drift report generation failed: {type(exc).__name__}: {exc}\n")
    print(json.dumps({name: {key: report.get(key) for key in (
        "drift_share", "dataset_drift", "number_of_drifted_columns", "drifted_columns", "test_names"
    )} for name, report in result.items() if isinstance(report, dict) and "drift_share" in report}, indent=2))


if __name__ == "__main__":
    main()
