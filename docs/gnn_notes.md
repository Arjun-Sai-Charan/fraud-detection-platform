# Graph and GNN notes (updated Phase 5b run)

The graph has transaction nodes and entity nodes for UID, `DeviceInfo`, `P_emaildomain`, and `addr1`. Each transaction connects to each non-null entity, and reverse edges are included. Labels are kept outside the graph and used only for loss and evaluation.

Transaction nodes use the same **52** numeric input features as the existing XGBoost model. Missing values are median-imputed and standardized using statistics fit on the 90,000 training transactions only. The two-layer heterogeneous GraphSAGE has a direct transaction-feature projection into the classifier path, so transaction logits use both the transaction's own features and messages from neighboring entities/transactions. It uses BCE with the train negative/positive ratio as `pos_weight`, dropout 0.2, and early stopping on validation PR-AUC.

## Training selection and convergence

The first implementation stopped after eight full-batch epochs and validation PR-AUC 0.0439. That run was underfit and its no-lift conclusion has been superseded. Phase 5b raises the limit to 300 epochs, patience to 30, and evaluates validation PR-AUC after each epoch. It logs per-epoch train loss and validation PR-AUC to MLflow and saves the traces in `results/gnn_training_curve.png` and `results/ablation.json`.

The requested grid is learning rates 0.003, 0.01, 0.03 and hidden sizes 64, 128, with dropout 0.2. The first candidate early-stopped and the second was truncated at epoch 130 by the 600-second cap; only hidden size 64 at learning rates 0.003 and 0.01 were evaluated. Thus this run is a **partial grid**. The selected setting is hidden size 64, learning rate 0.01, dropout 0.2, best epoch 121, validation PR-AUC 0.158862. Logistic regression on the same 52 features scored 0.083208 validation PR-AUC, so the selected GNN passed the requested sanity gate. The shuffled-edge control used the selected hyperparameters and scored 0.140724 validation PR-AUC. Its closeness to the real graph means the gain from real entity links remains uncertain.

`pyg-lib` and `torch-sparse` are absent in the pinned CPU environment; training is full-batch with no `NeighborLoader`. The train/validation/test snapshots contain 90,000 / 120,000 / 150,000 transactions. Training and evaluation took about ten minutes, with the selection grid time cap recorded as reached in the ablation result. The local MLflow file store registered model version 3. Server-mode MLflow remains UNVERIFIED until `make train-gnn` is run by the user.

## Graph construction and temporal limits

The entity degree cap is the 99.5th percentile of TRAIN degree by entity type. In the final snapshot, removed node/edge counts were:

| Entity type | Cap | Entity nodes | Hub nodes | Edges removed |
|---|---:|---:|---:|---:|
| UID | 99 | 24,084 | 187 | 46,898 |
| Device | 720 | 1,095 | 6 | 45,307 |
| Email domain | 32,627 | 59 | 1 | 55,834 |
| addr1 | 6,376 | 267 | 4 | 43,999 |

Train fitting uses train transactions only; validation scoring uses train plus validation features; test scoring uses train plus validation plus test features. Labels never become node inputs. However, within a validation or test snapshot, an earlier transaction can receive messages from entity-sharing transactions later in that same split. This feature-only message path makes offline scores somewhat optimistic versus a strictly incremental live graph.

## Results

Results below are from `results/ablation.json`. Test fraud base rate is 3.6667%, so interpret PR-AUC with that baseline. The validation-selected XGBoost blend weight is 1.0, meaning the blend is XGBoost alone.

| Model | Test PR-AUC | Test ROC-AUC |
|---|---:|---:|
| XGBoost | 0.553059 | 0.893220 |
| Real-graph GNN | 0.271741 | 0.766474 |
| Blend | 0.553059 | 0.893220 |
| Shuffled-edge GNN | 0.263502 | 0.811230 |

The real graph GNN improves over the shuffled control by a small PR-AUC margin on test, but remains far below XGBoost. It does not change the selected blend. Do not describe this as an overall graph-model lift.

The shuffled control independently permutes entity endpoints within relations and snapshots while preserving endpoint degree counts. The closeness of validation PR-AUC (0.15886 real vs 0.14072 shuffled) cautions against attributing the GNN's performance primarily to meaningful entity relationships.

The graph-feature XGBoost experiment computes strictly prior counts for UID, device, email domain and addr1, plus the prior number of distinct UIDs per device. It is a separate causal feature experiment; its test PR-AUC is 0.530860, below the base XGBoost 0.553059, so it is not selected.
