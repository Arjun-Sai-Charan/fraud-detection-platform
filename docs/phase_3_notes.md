# Phase 3: XGBoost and MLflow

## What runs

`make train-xgb` uses the default tracking URI `http://localhost:5000`. `make train-xgb-local` uses `file:./mlruns_local`, which is ignored by Git. Both targets invoke the same code and fixed CPU-only parameters, seed 42, single-threaded fitting, and stable feature ordering.

Training first asks Feast 0.40.1 for historical features using each `(transaction_id, event_timestamp)` pair. It joins only the target label, split, and numeric `transaction_dt` metadata back from `features.parquet`. In the local run this succeeded in **7.17 seconds**, so the parquet fallback was not used. If Feast errors or retrieval takes longer than five minutes, the code falls back to reading `features.parquet` and records the reason in the metrics JSON.

The model inputs are numeric features only. `transaction_id`, `event_timestamp`, `uid`, `isFraud`, and `split` are excluded; string category columns are represented by the encoded numeric columns generated in Phase 2. The validation and test labels are never model inputs.

The baselines are a seeded logistic regression pipeline with median imputation and standard scaling, and a random-ranking reference whose PR-AUC equals the split fraud rate. Two XGBoost models use the same settings and validation early stopping on `aucpr`: one with `scale_pos_weight=1`, and one with the training split's negative-to-positive ratio (**37.2328**). The run with the higher validation PR-AUC is registered as `fraud-xgb`.

Alert-budget results rank by score and use ascending `transaction_id` to break score ties, so each 1%, 2%, 3%, or 5% operating point flags exactly that share (rounded up for non-integral counts). The serving cutoff is the validation score at the 3% boundary, with the tie boundary recorded alongside it. That fixed validation cutoff is applied unchanged to test; the test alert share can differ from 3%.

## Results from the local file-store run

These results were produced by `make train-xgb-local`, tracked under `file:./mlruns_local`, and saved to `results/xgb_metrics.json`. The PR plot is `results/xgb_pr_curve.png` and includes validation and test base-rate lines. The registered model is local version **1** of `fraud-xgb`; its `model/` artifacts include `serving_threshold.json` and `feature_list.json`.

The run used **53 model features** and Feast historical retrieval. Random-reference PR-AUC equals the measured base rate: train **2.6156%**, validation **1.7200%**, test **3.6667%**. PR-AUC should be interpreted alongside these split-specific base rates; the higher test base rate raises the random reference and affects PR-AUC comparisons.

| Model | Validation PR-AUC | Validation ROC-AUC | Test PR-AUC | Test ROC-AUC |
|---|---:|---:|---:|---:|
| Logistic regression | 0.07513 | 0.76465 | 0.11651 | 0.78548 |
| XGBoost, no class weight | 0.43579 | 0.88621 | 0.54791 | 0.89221 |
| XGBoost, `scale_pos_weight=37.2328` | 0.39827 | 0.87827 | 0.52357 | 0.88304 |

The unweighted model had the higher validation PR-AUC and was registered. At each split's own alert budget, its precision and recall were:

| Alert budget | Validation precision | Validation recall | Test precision | Test recall |
|---:|---:|---:|---:|---:|
| 1% | 0.6200 | 0.3605 | 0.9133 | 0.2491 |
| 2% | 0.4083 | 0.4748 | 0.7417 | 0.4045 |
| 3% | 0.3100 | 0.5407 | 0.5844 | 0.4782 |
| 5% | 0.2053 | 0.5969 | 0.4320 | 0.5891 |

The validation-selected serving threshold is **0.06467335**, flagging **900 of 30,000 validation rows (3%)**. Its validation confusion counts are TP **279**, FP **621**, TN **28,863**, FN **237** (precision **0.3100**, recall **0.5407**). Applying the same score and tie cutoff to test flags **1,411 of 30,000 rows (4.7033%)**: TP **635**, FP **776**, TN **28,124**, FN **465** (precision **0.4500**, recall **0.5773**).

Validation and test PR-AUC differ alongside their different fraud rates: **1.72%** versus **3.67%**. PR-AUC depends on the positive-class prevalence, so the test value should not be read as a directly comparable improvement without considering its higher base rate. ROC-AUC is less sensitive to prevalence, but it also measures a different property.

## Verification status

- `make train-xgb-local`: completed both XGBoost runs, both baselines, PR curve and metrics output, and local `fraud-xgb` registration.
- Local model artifacts were inspected; the threshold and feature-list JSON files are next to the model files.
- `make test`: **10 passed**. Tests cover model-input exclusions and exact top-budget threshold selection on a tied-score toy array.
- **UNVERIFIED: server tracking mode.** The sandbox cannot reach `http://localhost:5000`. The HTTP server run remains unverified until you run `make train-xgb` and paste the output. No server-run results are claimed here.
