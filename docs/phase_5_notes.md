# Phase 5: Graph, GNN, and blend

## What was built

`make train-gnn-local` runs the local CPU workflow with `file:./mlruns_local`; `make train-gnn` targets the MLflow server at `http://localhost:5000`. Both use the same deterministic code, random seed 42, and fixed small GraphSAGE configuration. The real graph model is registered as `fraud-gnn` (local version 2). The shuffled-edge control is also logged to MLflow.

The graph includes transaction, uid, device, email-domain, and addr1 nodes, with forward and reverse edges. The current XGBoost artifact provides **52** numeric model inputs; `transaction_dt` remains metadata and is not an input. Transaction features are median-imputed and standardized using training rows only. Labels are never graph node inputs. Training uses only train transactions and labels; validation and test scores use cumulative train+validation and train+validation+test feature graphs respectively.

Neither `pyg-lib` nor `torch-sparse` is installed, so this environment uses full-batch graph computation without `NeighborLoader`. The train, validation, and test snapshot sizes are 90,000, 120,000, and 150,000 transactions. Real graph model fitting took 8.14 seconds; shuffled-edge fitting took 7.74 seconds. The full local pipeline, including graph preparation, scoring, the causal graph-feature experiment, MLflow logging, and registration, completed in about two minutes.

`results/ablation.json` has the complete run, split rates, validation blend grid, graph cap report, metrics, and alert-budget results. `artifacts/gnn/` contains `gnn_scores.parquet`, `rings.parquet`, `blend_config.json`, and the hub report; it is ignored by Git. `fraud-gnn` is registered in the local file store. MLflow server mode remains **UNVERIFIED** until you run `make train-gnn` in your terminal.

## Degree cap and temporal limits

The degree cap is the 99.5th percentile of TRAIN degree for each entity type. In the full snapshot, caps removed 187 uid nodes / 46,898 edges, 6 device nodes / 45,307 edges, 1 email-domain node / 55,834 edges, and 4 addr1 nodes / 43,999 edges. The per-type caps and remaining edges are in the `hub_cap` section of `results/ablation.json` and are explained in [gnn_notes.md](gnn_notes.md).

Labels never enter node features. Still, within a validation or test snapshot a transaction can receive feature-only messages from entity-sharing transactions later in the same split. Offline GNN scores may therefore be slightly optimistic compared with strictly live scoring. Ring assignment is also offline: it groups flagged test transactions into connected components. A streaming system would build ring membership incrementally.

## Test results

The test fraud base rate is **3.6667%**. PR-AUC should be considered alongside this base rate. The GNN had no useful lift: its validation PR-AUC was **0.04389**, compared with **0.41722** for XGBoost. The shuffled-edge validation PR-AUC was **0.04472**, effectively matching the real graph. On test, the real and shuffled graph PR-AUCs were **0.13955** and **0.13434**, while XGBoost scored **0.55306**. The validation blend grid selected XGBoost weight **1.0**, so the final blend is the XGBoost score. The graph did not add predictive value in this run.

| Test scorer | PR-AUC | ROC-AUC |
|---|---:|---:|
| XGBoost | 0.55306 | 0.89322 |
| GNN, real graph | 0.13955 | 0.75355 |
| Blend (XGBoost weight 1.0) | 0.55306 | 0.89322 |
| GNN, shuffled edges | 0.13434 | 0.75018 |

Precision / recall at each model's top alert budget:

| Test scorer | 1% P / R | 2% P / R | 3% P / R | 5% P / R |
|---|---:|---:|---:|---:|
| XGBoost | 0.8967 / 0.2445 | 0.7550 / 0.4118 | 0.6133 / 0.5018 | 0.4373 / 0.5964 |
| GNN, real graph | 0.2600 / 0.0709 | 0.2083 / 0.1136 | 0.1978 / 0.1618 | 0.1740 / 0.2373 |
| Blend | 0.8967 / 0.2445 | 0.7550 / 0.4118 | 0.6133 / 0.5018 | 0.4373 / 0.5964 |
| GNN, shuffled edges | 0.2733 / 0.0745 | 0.2417 / 0.1318 | 0.2044 / 0.1673 | 0.1773 / 0.2418 |

The validation top-3% blend threshold is **0.1192927361**, selecting 900 of 30,000 validation rows. Because the selected XGBoost blend weight is 1.0, this matches the Phase 3 XGBoost score threshold. On test, that fixed threshold flags 1,184 rows. Among them, **893 transactions** belong to **46 multi-transaction rings**; the rest have no multi-transaction ring assignment.

Since GNN did not improve validation PR-AUC and shuffled edges performed about the same, a compact causal graph-feature XGBoost experiment was also run. It added five prior-history features (four entity degree counts and distinct uids per device). With 57 features total, it reached validation PR-AUC **0.42811** (base XGBoost: 0.41722) but test PR-AUC **0.53086** and ROC-AUC **0.88562**, below base XGBoost's 0.55306 / 0.89322. The small validation gain did not carry to test, so these extra features are not selected.

## Serving and tests

The serving scorer reads precomputed GNN scores and ring assignments by transaction ID. When a GNN score exists, it returns `xgb_score`, `gnn_score`, the weighted `final_score`, and ring fields; the alert uses the saved blend threshold. If GNN artifacts or a score for that ID are absent, it falls back to the XGBoost threshold and score. The response schema is unchanged. A local scorer call using Feast's SQLite online store and the generated GNN artifacts was verified.

Focused tests cover blend math, ring grouping, hub-edge capping, train-only normalization, scorer fallback, and serving of saved GNN/ring artifacts. The full `.venv/bin/python -m pytest -q` run completed with **23 passed** (31 dependency deprecation warnings). Kafka and MLflow server integrations were not contacted in this environment.
