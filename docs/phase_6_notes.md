# Phase 6: Drift reports and candidate retraining

## What drift means here

Feature drift means the values being scored differ from the reference values the model was trained around. It is a warning to investigate data changes; it does not by itself show that model accuracy fell or that the fraud process changed. We also compare model score distributions, and report fraud-label rates separately because those labels are not available at the moment a live transaction is scored.

The reference is the **90,000 TRAIN rows** and the report compares the same 52 XGBoost input features on validation and test. Evidently 0.5.1 was inspected in the installed environment: this implementation uses `Report`, `DataDriftPreset`, `Report.run(reference_data=..., current_data=...)`, `Report.as_dict()`, and `Report.save_html()`. Drift JSON includes Evidently's observed share of drifted columns, drifted feature names, per-column test names, row counts, and fraud rates where labels are available.

## Thresholds and observed results

`src/common/config.py` sets `DRIFT_SHARE_THRESHOLD=0.30` and `MIN_ANALYST_FEEDBACK=25`, overridable with `FRAUD_DRIFT_SHARE_THRESHOLD` and `FRAUD_MIN_ANALYST_FEEDBACK`. A 30% feature threshold means at least 16 of 52 columns must be detected as drifted before the drift trigger fires; this reduces triggering on one or two isolated features. The 25-label trigger waits for a small batch of analyst feedback rather than reacting to one case. These are operational starting points, not values statistically optimized for this dataset.

| Report | Rows / comparison | Drift result | Fraud rates |
|---|---|---|---|
| Validation feature drift | 90,000 train vs 30,000 validation | **0.40385** share; 21/52 features drifted; trigger threshold exceeded. [results/drift/validation_data_drift.json](../results/drift/validation_data_drift.json) |
| Test feature drift | 90,000 train vs 30,000 test | **0.40385** share; 21/52 features drifted; trigger threshold exceeded. [results/drift/test_data_drift.json](../results/drift/test_data_drift.json) |
| Deliberately shifted test | Train vs 30,000 transformed test rows | **0.44231** share; 23/52 features drifted. TransactionAmt was multiplied by 2.5, both velocity counts by 2, and device-change was set to 1. [results/drift/shifted_test_data_drift.json](../results/drift/shifted_test_data_drift.json) |
| XGBoost score drift | 30,000 validation scores vs 30,000 test scores | **0.0** share; 0/1 score column drifted, so score-distribution drift did not fire. [results/drift/score_drift.json](../results/drift/score_drift.json) |

The label rate rises from **1.72% in validation to 3.67% in test**, while the feature-drift reports also detect 21 input columns in each window. The score-only report does not detect drift. These findings are not contradictory: Evidently tests the distribution of model scores, while the fraud rate uses labels that are unavailable during live scoring. Drift detection is a signal for review, not a substitute for outcome-based evaluation.

The feature reports used Jensen-Shannon distance and normalized Wasserstein distance, as selected by Evidently for the column types. The one-column score report used normalized Wasserstein distance. HTML files were saved under `results/drift/`; only `score_drift.html` is tracked because it is 2,981,195 bytes, below 3 MB. The three feature-report HTML files exceed 3 MB and remain ignored; all per-report JSON files and the combined `drift_summary.json` are tracked.

## Candidate retraining

`src/monitoring/retrain.py` reads the test drift JSON and feedback table. It triggers if test `drift_share` is **strictly greater than 0.30** or the SQLite feedback table contains at least **25 labels**. The current local run triggered on test feature drift at 0.40385; the serving database had zero feedback rows.

The local retraining run used `file:./mlruns_local`, tagged both XGBoost runs with the drift reason, registered `fraud-xgb` version **3**, and assigned it the `candidate` alias. The selected candidate's validation PR-AUC was **0.41722**. It remains a candidate: registry serving and export now prefer the `champion` alias, or the newest non-candidate version when no champion alias exists. They do not select the candidate implicitly. The exported local serving files were not overwritten.

Run commands from the repository root:

```bash
make drift
make retrain-local RETRAIN_ARGS=--dry-run
make retrain-local
make retrain-local RETRAIN_ARGS=--include-feedback
```

The server-mode equivalents are `make retrain RETRAIN_ARGS=--dry-run` and `make retrain`; server mode is **UNVERIFIED** until it is run against the user's MLflow service. A dry run writes `results/retrain_log.json` but does not call training or alter the model registry.

With `--include-feedback`, only resolved/closed/confirmed rows with recognized analyst labels (`fraud`, `confirmed_fraud`, `legit`, `legitimate`, `not_fraud`, or `benign`) are joined to features from the test replay. They are added to the training split. Because those test-replay rows then participate in training, that candidate is evaluated on validation only; its test metrics are deliberately omitted.

## Tests and limits

Phase 6 tests exercise Evidently drift share on synthetic data, a deliberately shifted slice, the below-threshold no-trigger case, dry-run no-training behavior, feedback-count triggering with a temporary SQLite database, feedback-label mapping, validation-only evaluation when feedback is included, and candidate alias exclusion from serving-version selection. Local run and tests do not verify the user's MLflow HTTP service or Kafka.
