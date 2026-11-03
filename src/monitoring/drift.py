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

from src.common.config import DRIFT_SHARE_THRESHOLD, PROCESSED_DIR, RESULTS_DIR, XGB_ARTIFACT_DIR

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
    report = Report(metrics=[DataDriftPreset(columns=columns, drift_share=drift_share_threshold)])
    report.run(reference_data=reference.loc[:, columns], current_data=current.loc[:, columns])
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


def build_shifted_test(test: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    shifted = test.copy()
    for column, multiplier in (
        ("TransactionAmt", 2.5),
        ("txn_count_last_1h", 2.0),
        ("txn_count_last_24h", 2.0),
    ):
        if column in feature_names:
            shifted[column] = pd.to_numeric(shifted[column], errors="coerce") * multiplier
    if "device_change_vs_previous" in feature_names:
        shifted["device_change_vs_previous"] = 1.0
    return shifted


def generate_reports(
    features_path: Path = FEATURES_PATH,
    output_dir: Path = DRIFT_DIR,
    drift_share_threshold: float = DRIFT_SHARE_THRESHOLD,
) -> dict[str, Any]:
    frame = pd.read_parquet(features_path)
    feature_names = _feature_names()
    if len(feature_names) != len(set(feature_names)):
        raise ValueError("Saved XGBoost feature list contains duplicate names")
    missing = sorted(set(feature_names) - set(frame.columns))
    if missing:
        raise ValueError(f"Saved feature columns missing from features parquet: {missing}")
    split_frames = {
        name: frame.loc[frame["split"] == name].sort_values("transaction_id", kind="mergesort").copy()
        for name in ("train", "validation", "test")
    }
    if any(part.empty for part in split_frames.values()):
        raise ValueError("Expected non-empty train, validation, and test windows")
    reference = split_frames["train"].loc[:, feature_names].apply(pd.to_numeric, errors="coerce")
    validation = split_frames["validation"].loc[:, feature_names].apply(pd.to_numeric, errors="coerce")
    test = split_frames["test"].loc[:, feature_names].apply(pd.to_numeric, errors="coerce")
    rates = {name: float(part["isFraud"].mean()) for name, part in split_frames.items()}
    results: dict[str, Any] = {"reference_split": "train", "feature_count": len(feature_names), "split_fraud_rates": rates}
    for name, current in (("validation_data_drift", validation), ("test_data_drift", test)):
        results[name] = make_report(
            reference, current, columns=feature_names, report_name=name, output_dir=output_dir,
            drift_share_threshold=drift_share_threshold,
            metadata={"reference_split": "train", "current_split": name.removesuffix("_data_drift"),
                      "reference_fraud_rate": rates["train"],
                      "current_fraud_rate": rates[name.removesuffix("_data_drift")]},
        )
    shifted = build_shifted_test(test, feature_names)
    results["shifted_test_data_drift"] = make_report(
        reference, shifted, columns=feature_names, report_name="shifted_test_data_drift",
        output_dir=output_dir, drift_share_threshold=drift_share_threshold,
        metadata={"reference_split": "train", "current_split": "deliberately_shifted_test",
                  "shifts": {"TransactionAmt": 2.5, "txn_count_last_1h": 2.0,
                             "txn_count_last_24h": 2.0, "device_change_vs_previous": "set to 1"}},
    )

    model = mlflow.xgboost.load_model(str(XGB_ARTIFACT_DIR / "model"))
    validation_scores = model.predict_proba(validation)[:, 1]
    test_scores = model.predict_proba(test)[:, 1]
    results["score_drift"] = make_report(
        pd.DataFrame({"xgb_score": validation_scores}),
        pd.DataFrame({"xgb_score": test_scores}), columns=["xgb_score"], report_name="score_drift",
        output_dir=output_dir, drift_share_threshold=drift_share_threshold,
        metadata={"reference_split": "validation", "current_split": "test",
                  "reference_fraud_rate": rates["validation"], "current_fraud_rate": rates["test"]},
    )
    summary_path = output_dir / "drift_summary.json"
    summary_path.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=FEATURES_PATH)
    parser.add_argument("--output-dir", type=Path, default=DRIFT_DIR)
    parser.add_argument("--threshold", type=float, default=DRIFT_SHARE_THRESHOLD)
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
