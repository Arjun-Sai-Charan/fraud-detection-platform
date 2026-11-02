# Graph and GNN notes

The graph has one transaction node for each Phase 1 transaction, plus entity nodes for `uid`, `DeviceInfo`, `P_emaildomain`, and `addr1`. A transaction connects to each non-null entity and each edge has a reverse edge. Labels are kept outside the graph and are used only in the training loss and evaluation.

Transaction nodes use the current local XGBoost model's saved list of **52** numeric features. (`transaction_dt` was removed from that list in the Phase 3/4 schema correction; it remains metadata.) Missing values are replaced by medians and columns are standardized using statistics from the 90,000 training rows only. Entity nodes start with a constant feature. The model is a two-layer, 64-hidden-unit GraphSAGE with one SAGEConv per relation, BCE loss weighted by the train negative-to-positive ratio, and validation PR-AUC early stopping.

Neither `pyg-lib` nor `torch-sparse` is installed for the local PyTorch 2.5.1 CPU environment. Training therefore uses **full-batch** graph operations, without `NeighborLoader`. This smallest model finished each fitting run in about 8 seconds (the full pipeline, including graph preparation, scoring, MLflow artifacts, and the additional XGBoost experiment, completed in about two minutes).

The degree cap for each entity type is the 99.5th percentile of that type's degree in TRAIN. The measured caps and final-snapshot removals were:

| Entity type | Degree cap | Entity nodes | Hub nodes removed | Edges removed |
|---|---:|---:|---:|---:|
| uid | 99 | 24,084 | 187 | 46,898 |
| device | 720 | 1,095 | 6 | 45,307 |
| email domain | 32,627 | 59 | 1 | 55,834 |
| addr1 | 6,376 | 267 | 4 | 43,999 |

Three cumulative snapshots enforce ordering: train fitting uses only 90,000 train transactions and their edges; validation scoring uses 120,000 train-plus-validation transactions; test scoring uses all 150,000. For each snapshot, hub decisions use only entity degrees observed up to that snapshot, with the cap itself fixed from train. Labels are never node inputs.

A remaining temporal caveat is that within a validation or test snapshot, a transaction can receive feature-only messages from entity-sharing transactions that occur later in the same split. This can make offline GNN scores slightly optimistic compared with a live, strictly incremental scorer.

The shuffled-edge control independently permutes entity endpoints within each relation and snapshot. This preserves the transaction and entity degree counts while disrupting which transactions share entities. Its validation PR-AUC was 0.04472, nearly the real graph's 0.04389. That indicates the observed GNN result does not depend on the learned transaction/entity links.

The graph-feature experiment separately computes strictly prior counts for uid, device, email domain, and addr1, plus the prior number of distinct uids per device. It does not use current or future transactions to compute these values.
