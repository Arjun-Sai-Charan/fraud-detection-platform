# Phase 6b: drift calibrated to natural change and candidate retraining

## What drift means

Drift is a change in feature distributions, not proof that accuracy fell or that fraud behavior changed. This phase uses Evidently 0.5.1 on the 52 XGBoost input features. Its API was checked in the installed environment (`Report`, `DataDriftPreset`, `Report.run`, `as_dict`, and `save_html`). Every report records the detected columns and the statistical test names. Columns that can accumulate or change as history grows (the `D*` values, transaction counts, `uid_prior_txn_count`, and `uid_is_hub`) are tagged `cumulative_by_construction`.

## Natural baseline and trigger

The 150,000-row timeline was divided chronologically into ten consecutive windows of 15,000 rows. Each of the nine windows after the first is compared with its immediately previous window. The observed drift-share distribution in [natural_baseline.json](../results/drift/natural_baseline.json) is mean **0.2158**, maximum **0.5000**, and p95 **0.4231**. The trigger rule is the observed natural maximum plus 0.10, capped at 1.0, giving **0.60**. `src/common/config.py` reads this threshold from the baseline file when it exists; `FRAUD_DRIFT_SHARE_THRESHOLD` can override it.

The validation-versus-train and test-versus-train shares are both **0.4038** (21/52 columns each). They are informational only: comparing a later period with all training history fires on ordinary time evolution. The actual most recent window versus its previous window has share **0.0769** (4/52), below 0.60, so real recent-window drift does **not** trigger. Score drift between validation and test is **0.0** (0/1 score columns); fraud label rates are separate outcomes, **1.72%** in validation and **3.67%** in test.

The deliberately shifted last-window demonstration reports **1.0000** (52/52) versus the previous real window, above threshold. It multiplies transaction amount by 2.5, doubles both velocity counts, sets device-change to 1, and applies large synthetic offsets to the remaining features. This is explicitly a broad synthetic stress test, not a plausible observed distribution. It is not trigger eligible. Only `recent_window_drift.json` is used as a drift trigger input. All vs-TRAIN reports are informational. JSON for the reports is tracked; feature HTML reports exceed 3 MB and are not tracked. The score HTML report is under 3 MB and is tracked.

## Expanded candidate retraining

A local run using `file:./mlruns_local` exercised candidate registration with the **synthetic stress report** as its explicit trigger input, because the real recent window did not trigger. Training used actual train+validation data, not shifted data. The champion was version **2**; its best iteration was **402**. Training on 120,000 rows used `round(402 * 120000 / 90000) = 536` estimators, with the champion's logged hyperparameters and no early stopping. Feast historical feature retrieval succeeded.

Candidate version **4** was registered with the `candidate` alias and the trigger reason. The serving resolver still selected version **2**; no `champion` alias exists yet, and no automatic promotion occurred. Version 2 is the newest non-candidate fallback. The explicit promotion command creates or moves the `champion` alias. For the requested final comparison only, both models were scored on the 30,000-row test split (fraud base rate **3.67%**):

| Model | Test PR-AUC | Test ROC-AUC | Precision / recall at 3% alerts |
|---|---:|---:|---:|
| Champion v2 | 0.55306 | 0.89322 | 0.61333 / 0.50182 |
| Candidate v4 | 0.59838 | 0.90012 | 0.64333 / 0.52636 |

Full budget metrics for 1%, 2%, 3%, and 5% are in [retrain_log.json](../results/retrain_log.json). Test was used for this final champion-versus-candidate comparison only; it did not choose hyperparameters. Promotion remains manual: inspect a candidate and run `make promote-candidate PROMOTE_ARGS="--version N"` (or `.venv/bin/python -m scripts.promote_candidate --version N`). `--dry-run` previews without changing aliases. The server MLflow run is **UNVERIFIED** because this sandbox cannot reach the user's MLflow service.

The feedback path remains available with `--include-feedback`. Confirmed analyst-labeled rows from the test replay are added to training; such a candidate is evaluated on validation only because feedback rows originate from test replay.

## Serving latency and checks

The 5,000-transaction benchmark was rerun after caching/filtering the repeated Feast feature-view deprecation warning. It measures Feast online lookup plus model inference, not Kafka: p50 **26.28 ms**, p95 **52.14 ms**, p99 **74.36 ms**, throughput **33.01 transactions/second**. Raw result: [latency.json](../results/latency.json). The change filters only the exact repeated Feast warning; other warnings remain visible.

Tests cover synthetic natural windows and threshold selection, shifted-window detection, no-trigger behavior, dry-run immutability, expanded row/estimator arithmetic, feedback triggering, candidate non-promotion, and promotion dry-run. Local registry behavior was exercised; server-mode retraining and promotion are **UNVERIFIED**.
