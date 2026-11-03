# Phase 5: Graph, GNN, and blend (updated Phase 5b run)

## What changed

The original GNN run stopped after only eight full-batch updates and reached validation PR-AUC 0.0439. That was too early to support the previous no-lift conclusion. This retraining adds a direct transaction-feature projection to the classifier path, trains up to 300 epochs with patience 30, evaluates validation PR-AUC each epoch, and tests learning rates 0.003, 0.01, and 0.03 with hidden sizes 64 and 128. Dropout is 0.2. Each epoch's training loss and validation PR-AUC are logged to MLflow. The selected configuration and epoch traces are in `results/ablation.json`; the plotted traces are in `results/gnn_training_curve.png`.

The local run selected hidden size 64 and learning rate 0.01, with best validation PR-AUC **0.15886** at epoch 121. That exceeds logistic regression's validation PR-AUC **0.08321** from `results/xgb_metrics.json`, and improves on the prior 0.0439 result. The first grid candidate early-stopped; the second was truncated at epoch 130 by the 600-second training cap, so four combinations (including hidden size 128) were not evaluated. This is a partial grid result, not a claim that the best possible setting was found. The shuffled-edge model was retrained with the selected hyperparameters; its validation PR-AUC was **0.14072**.

The shuffled result is close to the real graph result, so the evidence for a strong relational-structure benefit remains weak. On test, GNN PR-AUC is **0.27174** versus **0.26350** for shuffled edges; both remain below XGBoost at **0.55306**. Test was used only for final reporting, not grid or blend selection.

## Model and graph

`make train-gnn-local` runs against `file:./mlruns_local`; `make train-gnn` targets `http://localhost:5000`. The local run registered `fraud-gnn` version **3**. Server-mode training remains **UNVERIFIED** until `make train-gnn` is run in the user's terminal.

The heterogeneous graph contains transaction, UID, device, email-domain, and addr1 nodes with forward and reverse edges. Transaction nodes use the same 52 numeric inputs as the existing XGBoost model. Missing values are median-imputed and standardized using training rows only. The transaction classifier includes a direct projection of its own feature vector plus the two GraphSAGE message-passing layers. Labels are not graph node inputs.

Neither `pyg-lib` nor `torch-sparse` is installed, so the run uses full-batch graph computation without `NeighborLoader`. Train, validation and test snapshots contain 90,000, 120,000 and 150,000 transactions. The run took approximately 10 minutes overall and hit the configured 600-second cap during the validation grid; see `time_cap_hit` and `gnn_grid_runs` in the saved ablation JSON for exact detail.

The graph degree cap remains the 99.5th percentile of TRAIN degree by entity type. Saved removal counts are: UID 187 nodes / 46,898 edges; device 6 / 45,307; email domain 1 / 55,834; addr1 4 / 43,999. Rings are assigned offline to flagged test transactions, not incrementally as a streaming system would do.

## Test comparison

The test fraud base rate is 3.6667%; PR-AUC is shown with that base rate in mind. These values are from the current `results/ablation.json` run.

| Test scorer | PR-AUC | ROC-AUC |
|---|---:|---:|
| XGBoost | 0.55306 | 0.89322 |
| GNN, real graph | 0.27174 | 0.76647 |
| Blend | 0.55306 | 0.89322 |
| GNN, shuffled edges | 0.26350 | 0.81123 |

The validation blend search still selects XGBoost weight **1.0**, so the final blend is XGBoost alone. This training fixes the underfit evidence but does not establish that the graph improves the final model.

| Test scorer | 1% precision / recall | 2% precision / recall | 3% precision / recall | 5% precision / recall |
|---|---:|---:|---:|---:|
| XGBoost and blend | 0.8967 / 0.2445 | 0.7550 / 0.4118 | 0.6133 / 0.5018 | 0.4373 / 0.5964 |
| GNN, real graph | 0.5700 / 0.1555 | 0.4333 / 0.2364 | 0.3411 / 0.2791 | 0.2547 / 0.3473 |
| GNN, shuffled edges | 0.5033 / 0.1373 | 0.3867 / 0.2109 | 0.3200 / 0.2618 | 0.2707 / 0.3691 |

The blend threshold remains the validation top-3% boundary, **0.1192927361**, selecting 900/30,000 validation rows. The fixed threshold flags 1,184 test rows; 893 are assigned to 46 multi-transaction rings. The ring membership is an offline test analysis.

The causal graph-feature XGBoost experiment is unchanged: five strictly prior entity-history features produced validation PR-AUC 0.42811 and test PR-AUC 0.53086, below the original XGBoost test PR-AUC. Those features are not selected.

## Limitations and verification

Within each validation/test snapshot, earlier transactions can receive feature-only messages from transactions later in the same split. No labels enter the graph, but this makes offline scoring somewhat optimistic compared with live incremental scoring. Graph cap, normalization, ring and scorer behavior are covered by the Phase 5 tests.

The local-file-store run completed and wrote scores, rings, blend configuration, updated ablation metrics and a training curve. MLflow server mode, Kafka integration and the final live scoring path remain unverified in this environment.
