"""Train a small full-batch hetero GraphSAGE and validation-selected score blend."""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import random
import time
from pathlib import Path
from typing import Any

import mlflow
import mlflow.pytorch
import mlflow.xgboost
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import HeteroConv, SAGEConv

from src.common.config import PROCESSED_DIR, RESULTS_DIR, XGB_ARTIFACT_DIR
from src.models.blend import (
    BLEND_GRID,
    blend_scores,
    choose_blend_weight,
    connected_rings,
    evaluate_scores,
    fit_top3_threshold,
)
from src.models.build_graph import (
    ALL_RELATIONS,
    ENTITY_COLUMNS,
    FEATURES_PATH,
    GraphBundle,
    build_graph,
    _entity_key,
    _read_rows,
)

SEED = 42
TRACKING_EXPERIMENT = "fraud_graphsage_phase5"
MODEL_NAME = "fraud-gnn"
ARTIFACT_DIR = Path("artifacts/gnn")
SCORES_PATH = ARTIFACT_DIR / "gnn_scores.parquet"
RINGS_PATH = ARTIFACT_DIR / "rings.parquet"
ABLATION_PATH = RESULTS_DIR / "ablation.json"
HIDDEN_SIZE = 64
MAX_EPOCHS = 12
PATIENCE = 3


class HeteroSAGE(nn.Module):
    def __init__(self, hidden_size: int = HIDDEN_SIZE, dropout: float = 0.2):
        super().__init__()
        self.conv1 = HeteroConv(
            {relation: SAGEConv((-1, -1), hidden_size) for relation in ALL_RELATIONS},
            aggr="sum",
        )
        self.conv2 = HeteroConv(
            {relation: SAGEConv((hidden_size, hidden_size), hidden_size) for relation in ALL_RELATIONS},
            aggr="sum",
        )
        self.dropout = float(dropout)
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, graph) -> torch.Tensor:
        features = self.conv1(graph.x_dict, graph.edge_index_dict)
        features = {
            node_type: F.dropout(F.relu(value), p=self.dropout, training=self.training)
            for node_type, value in features.items()
        }
        features = self.conv2(features, graph.edge_index_dict)
        return self.classifier(features["transaction"]).squeeze(-1)


def set_deterministic(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(2)


def _predict(model: HeteroSAGE, graph, start: int, end: int) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits = model(graph)[start:end]
        return torch.sigmoid(logits).cpu().numpy().astype(np.float64)


def _train_graph_model(
    train_graph,
    validation_graph,
    labels: np.ndarray,
    train_count: int,
    validation_end: int,
    *,
    seed: int,
    run_name: str,
) -> tuple[HeteroSAGE, np.ndarray, dict[str, Any]]:
    set_deterministic(seed)
    model = HeteroSAGE()
    # Initialize lazy GraphSAGE weights before the optimizer is created.
    model.eval()
    with torch.no_grad():
        model(train_graph)
    model.train()
    y = torch.as_tensor(labels, dtype=torch.float32)
    positives = float(y[:train_count].sum())
    negatives = float(train_count - positives)
    if positives == 0:
        raise ValueError("Training split has no positive labels")
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(negatives / positives))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-4)
    val_y = labels[train_count:validation_end]
    best_score = -1.0
    best_state = None
    best_epoch = 0
    wait = 0
    started = time.perf_counter()
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(train_graph)[:train_count]
        loss = criterion(logits, y[:train_count])
        loss.backward()
        optimizer.step()

        validation_scores = _predict(model, validation_graph, train_count, validation_end)
        val_pr_auc = float(average_precision_score(val_y, validation_scores))
        if val_pr_auc > best_score + 1e-7:
            best_score = val_pr_auc
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
        if wait >= PATIENCE:
            break
    if best_state is None:
        raise RuntimeError("GNN training did not produce a checkpoint")
    model.load_state_dict(best_state)
    elapsed = time.perf_counter() - started
    info = {
        "run_name": run_name,
        "epochs_run": epoch,
        "best_epoch": best_epoch,
        "validation_pr_auc": best_score,
        "training_seconds": elapsed,
        "positive_weight": negatives / positives,
    }
    return model, validation_scores if best_epoch == epoch else _predict(model, validation_graph, train_count, validation_end), info


def _raw_xgb_scores(bundle: GraphBundle, features_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xgb_model = mlflow.xgboost.load_model(str(XGB_ARTIFACT_DIR / "model"))
    features = pd.read_parquet(features_path).set_index("transaction_id")
    matrix = features.loc[bundle.transaction_ids, bundle.feature_names].apply(pd.to_numeric, errors="coerce")
    scores = np.asarray(xgb_model.predict_proba(matrix)[:, 1], dtype=np.float64)
    return (
        scores[:bundle.train_count],
        scores[bundle.train_count:bundle.validation_end],
        scores[bundle.validation_end:],
    )


def _graph_derived_features(bundle: GraphBundle) -> pd.DataFrame:
    """Compute strictly prior entity counts and prior distinct UIDs per device."""
    raw, _ = _read_rows()
    counts: dict[str, dict[str, int]] = {name: {} for name in ENTITY_COLUMNS}
    device_uid_sets: dict[str, set[str]] = {}
    values: dict[str, list[float]] = {
        "uid_prior_degree": [],
        "device_prior_degree": [],
        "email_domain_prior_degree": [],
        "addr1_prior_degree": [],
        "device_prior_distinct_uids": [],
    }
    for row in raw.itertuples(index=False):
        row_values = {name: _entity_key(getattr(row, column)) for name, column in ENTITY_COLUMNS.items()}
        uid = row_values["uid"]
        device = row_values["device"]
        for name, output in (
            ("uid", "uid_prior_degree"),
            ("device", "device_prior_degree"),
            ("email_domain", "email_domain_prior_degree"),
            ("addr1", "addr1_prior_degree"),
        ):
            key = row_values[name]
            values[output].append(float(counts[name].get(key, 0)) if key is not None else 0.0)
        values["device_prior_distinct_uids"].append(
            float(len(device_uid_sets.get(device, set()))) if device is not None else 0.0
        )
        for name, key in row_values.items():
            if key is not None:
                counts[name][key] = counts[name].get(key, 0) + 1
        if device is not None and uid is not None:
            device_uid_sets.setdefault(device, set()).add(uid)
    frame = pd.DataFrame(values, dtype=np.float32)
    frame.insert(0, "transaction_id", bundle.transaction_ids)
    return frame


def _train_graph_xgb_experiment(bundle: GraphBundle, features_path: Path) -> dict[str, Any]:
    from xgboost import XGBClassifier

    base = pd.read_parquet(features_path).set_index("transaction_id").loc[
        bundle.transaction_ids, bundle.feature_names
    ].apply(pd.to_numeric, errors="coerce")
    extra = _graph_derived_features(bundle).set_index("transaction_id")
    x = pd.concat([base.reset_index(drop=True), extra.reset_index(drop=True)], axis=1)
    train_x, val_x, test_x = x.iloc[:bundle.train_count], x.iloc[bundle.train_count:bundle.validation_end], x.iloc[bundle.validation_end:]
    y_train = bundle.labels[:bundle.train_count]
    model = XGBClassifier(
        objective="binary:logistic", eval_metric="aucpr", n_estimators=400,
        early_stopping_rounds=30, max_depth=6, learning_rate=0.05,
        min_child_weight=5, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        tree_method="hist", device="cpu", n_jobs=1, random_state=SEED,
        scale_pos_weight=1.0, verbosity=0,
    )
    model.fit(train_x, y_train, eval_set=[(val_x, bundle.labels[bundle.train_count:bundle.validation_end])], verbose=False)
    validation_score = model.predict_proba(val_x)[:, 1]
    score = model.predict_proba(test_x)[:, 1]
    result = {
        "feature_count": int(x.shape[1]),
        "extra_features": list(extra.columns),
        "best_iteration": int(model.best_iteration),
        "validation": evaluate_scores(
            bundle.labels[bundle.train_count:bundle.validation_end],
            validation_score,
            bundle.transaction_ids[bundle.train_count:bundle.validation_end],
        ),
        "test": evaluate_scores(bundle.labels[bundle.validation_end:], score, bundle.transaction_ids[bundle.validation_end:]),
    }
    with mlflow.start_run(run_name="xgboost_causal_graph_features"):
        mlflow.set_tags({"model_family": "xgboost", "role": "graph_feature_experiment"})
        mlflow.log_params({"seed": SEED, "causal_features": len(extra.columns), "best_iteration": int(model.best_iteration)})
        mlflow.log_metrics({
            "validation_pr_auc": result["validation"]["pr_auc"],
            "validation_roc_auc": result["validation"]["roc_auc"],
            "test_pr_auc": result["test"]["pr_auc"],
            "test_roc_auc": result["test"]["roc_auc"],
        })
        mlflow.log_dict(result, "graph_feature_metrics.json")
    return result


def train(tracking_uri: str = "http://localhost:5000", features_path: Path = FEATURES_PATH) -> dict[str, Any]:
    set_deterministic()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(TRACKING_EXPERIMENT)
    bundle = build_graph(features_path)
    train_graph = bundle.snapshot(bundle.train_count)
    validation_graph = bundle.snapshot(bundle.validation_end)
    test_graph = bundle.snapshot(len(bundle.transaction_ids))
    shuffled_train = bundle.snapshot(bundle.train_count, shuffled=True)
    shuffled_validation = bundle.snapshot(bundle.validation_end, shuffled=True)
    shuffled_test = bundle.snapshot(len(bundle.transaction_ids), shuffled=True)

    y_val = bundle.labels[bundle.train_count:bundle.validation_end]
    y_test = bundle.labels[bundle.validation_end:]
    ids_val = bundle.transaction_ids[bundle.train_count:bundle.validation_end]
    ids_test = bundle.transaction_ids[bundle.validation_end:]
    xgb_train_scores, xgb_val_scores, xgb_test_scores = _raw_xgb_scores(bundle, features_path)

    run_summaries = {}
    predictions = {}
    trained_models = {}
    for name, graphs in (
        ("real_graph", (train_graph, validation_graph, test_graph)),
        ("shuffled_edges", (shuffled_train, shuffled_validation, shuffled_test)),
    ):
        model, _, info = _train_graph_model(
            graphs[0], graphs[1], bundle.labels, bundle.train_count, bundle.validation_end,
            seed=SEED, run_name=name,
        )
        val_scores = _predict(model, graphs[1], bundle.train_count, bundle.validation_end)
        test_scores = _predict(model, graphs[2], bundle.validation_end, len(bundle.transaction_ids))
        predictions[name] = {"validation": val_scores, "test": test_scores}
        trained_models[name] = model
        run_summaries[name] = info
        with mlflow.start_run(run_name=f"graphsage_{name}"):
            mlflow.set_tags({"model_family": "heterogeneous_graphsage", "graph_variant": name})
            mlflow.log_params({
                "seed": SEED, "hidden_size": HIDDEN_SIZE, "layers": 2,
                "max_epochs": MAX_EPOCHS, "patience": PATIENCE,
                "full_batch": True, "neighbor_loader": False,
                "positive_weight": info["positive_weight"], "hub_cap_percentile": 99.5,
            })
            mlflow.log_metrics({
                "validation_pr_auc": float(average_precision_score(y_val, val_scores)),
                "test_pr_auc": float(average_precision_score(y_test, test_scores)),
                "test_roc_auc": float(evaluate_scores(y_test, test_scores, ids_test)["roc_auc"]),
                "best_epoch": float(info["best_epoch"]),
                "training_seconds": info["training_seconds"],
            })
            if name == "real_graph":
                mlflow.pytorch.log_model(model, artifact_path="model")
                mlflow.log_dict(bundle.normalization, "train_normalization.json")
                mlflow.log_dict(bundle.hub_report, "hub_report.json")

    blend_weight, weight_scores = choose_blend_weight(y_val, xgb_val_scores, predictions["real_graph"]["validation"])
    val_blend = blend_scores(xgb_val_scores, predictions["real_graph"]["validation"], blend_weight)
    test_blend = blend_scores(xgb_test_scores, predictions["real_graph"]["test"], blend_weight)
    blend_threshold = fit_top3_threshold(val_blend, ids_val)
    test_flags = (test_blend > blend_threshold["threshold"]) | (
        (test_blend == blend_threshold["threshold"])
        & (ids_test <= blend_threshold["tie_break_transaction_id_max"])
    )
    test_rings = connected_rings(
        ids_test,
        test_flags,
        {name: values[bundle.validation_end:] for name, values in bundle.entity_keys.items()},
        bundle.non_hub_keys,
    )

    result = {
        "tracking_uri": tracking_uri,
        "registered_model_name": MODEL_NAME,
        "feature_count": len(bundle.feature_names),
        "feature_list": bundle.feature_names,
        "feature_source": "Phase 2 features.parquet aligned to Phase 1 split parquets",
        "pyg_lib_installed": importlib.util.find_spec("pyg_lib") is not None,
        "torch_sparse_installed": importlib.util.find_spec("torch_sparse") is not None,
        "training_mode": "full_batch_no_neighborloader",
        "temporal_snapshots": {
            "train_transactions": bundle.train_count,
            "validation_snapshot_transactions": bundle.validation_end,
            "test_snapshot_transactions": len(bundle.transaction_ids),
        },
        "split_base_rates": {
            "train": float(bundle.labels[:bundle.train_count].mean()),
            "validation": float(y_val.mean()),
            "test": float(y_test.mean()),
        },
        "hub_cap": {"selection": "99.5th percentile of TRAIN entity degree by type", "by_entity_type": bundle.hub_report},
        "train_normalization": bundle.normalization,
        "gnn_runs": run_summaries,
        "validation_blend_grid_pr_auc": {str(key): value for key, value in weight_scores.items()},
        "blend_weight_xgb": blend_weight,
        "blend_threshold": blend_threshold,
        "test_models": {
            "xgboost": evaluate_scores(y_test, xgb_test_scores, ids_test),
            "gnn": evaluate_scores(y_test, predictions["real_graph"]["test"], ids_test),
            "blend": evaluate_scores(y_test, test_blend, ids_test),
            "shuffled_edge_gnn": evaluate_scores(y_test, predictions["shuffled_edges"]["test"], ids_test),
        },
        "ring_summary": {
            "flagged_test_transactions": int(test_flags.sum()),
            "transactions_assigned_to_multi_transaction_rings": int(sum(row["ring_id"] is not None for row in test_rings)),
            "ring_count": int(len({row["ring_id"] for row in test_rings if row["ring_id"] is not None})),
        },
        "temporal_caveat": "Within a validation/test snapshot, a transaction can receive feature-only messages from entity-sharing transactions later in that same split; labels are never node inputs, but offline scores may be slightly optimistic versus live scoring.",
        "ring_caveat": "Ring components are assigned offline over flagged test transactions; streaming deployment would update components incrementally.",
    }

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    score_rows = []
    for split, ids, gnn_scores in (
        ("validation", ids_val, predictions["real_graph"]["validation"]),
        ("test", ids_test, predictions["real_graph"]["test"]),
    ):
        score_rows.extend({"transaction_id": int(tx_id), "split": split, "gnn_score": float(score)} for tx_id, score in zip(ids, gnn_scores))
    pd.DataFrame(score_rows).to_parquet(SCORES_PATH, index=False)
    pd.DataFrame(test_rings).to_parquet(RINGS_PATH, index=False)
    config = {"blend_weight_xgb": blend_weight, **blend_threshold}
    (ARTIFACT_DIR / "blend_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    (ARTIFACT_DIR / "hub_report.json").write_text(json.dumps(bundle.hub_report, indent=2) + "\n", encoding="utf-8")
    ABLATION_PATH.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    # If the graph does not improve validation PR-AUC over XGBoost, also try a
    # compact causal entity-history feature experiment as the requested fallback.
    xgb_val_pr = float(average_precision_score(y_val, xgb_val_scores))
    gnn_val_pr = float(average_precision_score(y_val, predictions["real_graph"]["validation"]))
    no_gnn_lift = gnn_val_pr <= xgb_val_pr
    if no_gnn_lift:
        result["graph_derived_xgb_experiment"] = _train_graph_xgb_experiment(bundle, features_path)
        result["graph_derived_xgb_experiment"]["computed_causally"] = True
        result["graph_derived_xgb_experiment"]["validation_gnn_lift"] = gnn_val_pr - xgb_val_pr
        ABLATION_PATH.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    real_runs = mlflow.search_runs(
        experiment_ids=[mlflow.get_experiment_by_name(TRACKING_EXPERIMENT).experiment_id],
        filter_string="tags.graph_variant = 'real_graph'",
        order_by=["start_time DESC"],
        max_results=1,
    )
    if real_runs.empty:
        raise RuntimeError("Could not find the real-graph MLflow run for registration")
    best_run_id = str(real_runs.iloc[0]["run_id"])
    registered = mlflow.register_model(f"runs:/{best_run_id}/model", MODEL_NAME)
    result["registered_model_version"] = int(registered.version)
    ABLATION_PATH.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracking-uri", default="http://localhost:5000")
    parser.add_argument("--features", type=Path, default=FEATURES_PATH)
    args = parser.parse_args()
    try:
        result = train(args.tracking_uri, args.features)
    except Exception as exc:
        parser.exit(1, f"GNN training failed: {type(exc).__name__}: {exc}\n")
    print(json.dumps({
        "tracking_uri": result["tracking_uri"],
        "feature_count": result["feature_count"],
        "gnn_runs": result["gnn_runs"],
        "blend_weight_xgb": result["blend_weight_xgb"],
        "blend_threshold": result["blend_threshold"],
        "test_models": result["test_models"],
        "ring_summary": result["ring_summary"],
        "registered_model_version": result.get("registered_model_version"),
    }, indent=2))


if __name__ == "__main__":
    main()
