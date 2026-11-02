# Phase 3: XGBoost and MLflow

## How training works

`make train-xgb-local` runs the deterministic training flow against `file:./mlruns_local`; `make train-xgb` uses the MLflow HTTP server at `http://localhost:5000`. Training requests historical features from Feast for each transaction and joins only the fraud label and split metadata from `features.parquet`. This local run used Feast historical retrieval. If retrieval fails or takes longer than five minutes, the code uses the parquet fallback and records the reason.

Model inputs are numeric features only. IDs, `uid`, timestamps and labels are excluded. `transaction_dt` remains available as time metadata, but is deliberately excluded from model inputs to avoid mixing Feast's online field with separately joined metadata. Before training/registration, the code checks that input feature names are unique and that every input is in the Feast view's online schema. The saved feature list has one entry per input.

The baselines are a median-imputed, scaled logistic regression and a random-ranking reference whose PR-AUC equals each split's fraud rate. Two XGBoost models use the same fixed CPU settings and seed, with validation early stopping on `aucpr`; one is unweighted and one uses the training negative-to-positive ratio. The model with higher validation PR-AUC is registered. A serving threshold is selected to flag the top 3% of validation rows, with transaction ID used to settle score ties.

## Latest local run

The run below was produced by `make train-xgb-local`, saved to `results/xgb_metrics.json` and `results/xgb_pr_curve.png`, and registered in the local file store as version **2** of `fraud-xgb`. The **previous 53-feature run and its metrics are superseded** by this run's 52-feature inputs. Server-mode training remains **UNVERIFIED** until the user runs `make train-xgb` and shares its output.

The split fraud rates are train **2.6156%**, validation **1.7200%**, and test **3.6667%**. PR-AUC is reported with each split's base rate because prevalence differs.

| Model | Validation PR-AUC | Validation ROC-AUC | Test PR-AUC | Test ROC-AUC |
|---|---:|---:|---:|---:|
| Logistic regression | 0.08321 | 0.76657 | 0.15682 | 0.79433 |
| XGBoost, no class weight | 0.41722 | 0.88083 | 0.55306 | 0.89322 |
| XGBoost, `scale_pos_weight=37.2328` | 0.34607 | 0.87909 | 0.50755 | 0.88357 |

The unweighted model was selected. Its precision/recall at each split's own alert budget:

| Alert budget | Validation precision | Validation recall | Test precision | Test recall |
|---:|---:|---:|---:|---:|
| 1% | 0.6033 | 0.3508 | 0.8967 | 0.2445 |
| 2% | 0.3883 | 0.4516 | 0.7550 | 0.4118 |
| 3% | 0.2933 | 0.5116 | 0.6133 | 0.5018 |
| 5% | 0.2113 | 0.6143 | 0.4373 | 0.5964 |

The validation threshold is **0.11929274** and flags 900 of 30,000 validation rows. On validation, TP=264, FP=636, TN=28,848, FN=252. Applying the same threshold to test flags 1,184 of 30,000 rows: TP=610, FP=574, TN=28,326, FN=490. Test alert share differs from 3% because the threshold was fixed on validation.

Validation and test PR-AUC differ in part because the fraud rates differ (1.72% versus 3.67%); the two values should not be interpreted as a direct like-for-like model improvement.

## Verification

- `make train-xgb-local`: completed Feast retrieval, both baselines, both XGBoost runs, metrics/PR plot, and local model registration.
- `.venv/bin/python -m pytest -q`: **17 passed** (31 dependency deprecation warnings). Focused Phase 3 tests cover input exclusions, unique names, Feast schema membership, and alert-budget tie handling.
- **UNVERIFIED: MLflow server mode.** The local file-store run is verified; HTTP server training was not run here.
