"""Build cumulative, hub-capped transaction/entity graphs for Phase 5."""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

from src.common.config import PROCESSED_DIR, XGB_ARTIFACT_DIR

SPLIT_FILES = {
    "train": PROCESSED_DIR / "train.parquet",
    "validation": PROCESSED_DIR / "validation.parquet",
    "test": PROCESSED_DIR / "test.parquet",
}
FEATURES_PATH = PROCESSED_DIR / "features.parquet"
ENTITY_COLUMNS = {
    "uid": "uid",
    "device": "DeviceInfo",
    "email_domain": "P_emaildomain",
    "addr1": "addr1",
}
RELATIONS = tuple(("transaction", f"to_{entity}", entity) for entity in ENTITY_COLUMNS)
REVERSE_RELATIONS = tuple((entity, f"rev_to_{entity}", "transaction") for entity in ENTITY_COLUMNS)
ALL_RELATIONS = RELATIONS + REVERSE_RELATIONS


def cap_entity_edges(
    transaction_indices: np.ndarray,
    entity_indices: np.ndarray,
    max_degree: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Remove edges incident to entities whose degree exceeds ``max_degree``."""
    tx = np.asarray(transaction_indices, dtype=np.int64)
    entity = np.asarray(entity_indices, dtype=np.int64)
    if tx.shape != entity.shape:
        raise ValueError("transaction and entity edge arrays must have equal shapes")
    degrees = np.bincount(entity) if len(entity) else np.zeros(0, dtype=np.int64)
    keep = degrees[entity] <= int(max_degree) if len(entity) else np.zeros(0, dtype=bool)
    return tx[keep], entity[keep], keep


def normalize_train_only(values: np.ndarray, train_count: int) -> tuple[np.ndarray, dict[str, list[float]]]:
    """Median-impute and standardize columns using only the leading train rows."""
    matrix = np.asarray(values, dtype=np.float64).copy()
    if matrix.ndim != 2 or not 0 < train_count <= len(matrix):
        raise ValueError("values must be a matrix and train_count must select at least one row")
    train = matrix[:train_count]
    medians = np.nanmedian(train, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    matrix = np.where(np.isnan(matrix), medians[None, :], matrix)
    means = matrix[:train_count].mean(axis=0)
    stds = matrix[:train_count].std(axis=0)
    stds = np.where(stds > 0, stds, 1.0)
    normalized = (matrix - means[None, :]) / stds[None, :]
    stats = {"median": medians.tolist(), "mean": means.tolist(), "std": stds.tolist()}
    return normalized.astype(np.float32), stats


def _entity_key(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return None if text.lower() in {"", "nan", "none", "<na>", "null"} else text


def _read_rows() -> tuple[pd.DataFrame, list[str]]:
    splits = []
    labels = []
    for split, path in SPLIT_FILES.items():
        frame = pd.read_parquet(path)
        frame["split"] = split
        splits.append(frame)
        labels.extend([split] * len(frame))
    raw = pd.concat(splits, ignore_index=True)
    raw["transaction_id"] = raw["TransactionID"].astype(np.int64)
    return raw, labels


@dataclass
class GraphBundle:
    transaction_ids: np.ndarray
    labels: np.ndarray
    splits: np.ndarray
    feature_names: list[str]
    transaction_x: torch.Tensor
    entity_keys: dict[str, list[str | None]]
    entity_key_to_index: dict[str, dict[str, int]]
    non_hub_keys: dict[str, set[str]]
    train_count: int
    validation_end: int
    normalization: dict[str, list[float]]
    hub_report: dict[str, dict[str, int | float]]
    max_degree: dict[str, int]

    def snapshot(self, cutoff: int, *, shuffled: bool = False, seed: int = 42) -> HeteroData:
        if not 0 < cutoff <= len(self.transaction_ids):
            raise ValueError("snapshot cutoff must include at least one transaction")
        data = HeteroData()
        data["transaction"].x = self.transaction_x[:cutoff]

        rng = np.random.default_rng(seed)
        for entity_type in ENTITY_COLUMNS:
            assignments = self.entity_keys[entity_type][:cutoff]
            degrees = Counter(key for key in assignments if key is not None)
            eligible_keys = [key for key, degree in degrees.items() if degree <= self.max_degree[entity_type]]
            key_to_index = {key: idx for idx, key in enumerate(eligible_keys)}
            data[entity_type].x = torch.ones((len(key_to_index), 1), dtype=torch.float32)
            txs, entities = [], []
            for tx_index, key in enumerate(assignments):
                if key is not None and key in key_to_index:
                    txs.append(tx_index)
                    entities.append(key_to_index[key])
            tx_array = np.asarray(txs, dtype=np.int64)
            ent_array = np.asarray(entities, dtype=np.int64)
            tx_array, ent_array, _ = cap_entity_edges(tx_array, ent_array, self.max_degree[entity_type])
            if shuffled and len(ent_array) > 1:
                ent_array = rng.permutation(ent_array)
            forward = torch.tensor(np.stack([tx_array, ent_array]), dtype=torch.long)
            reverse = torch.tensor(np.stack([ent_array, tx_array]), dtype=torch.long)
            data[("transaction", f"to_{entity_type}", entity_type)].edge_index = forward
            data[(entity_type, f"rev_to_{entity_type}", "transaction")].edge_index = reverse
        return data


def build_graph(
    features_path: Path = FEATURES_PATH,
    feature_list_path: Path = XGB_ARTIFACT_DIR / "feature_list.json",
    *,
    hub_percentile: float = 99.5,
) -> GraphBundle:
    if not 0 < hub_percentile <= 100:
        raise ValueError("hub_percentile must be in (0, 100]")
    raw, _ = _read_rows()
    features = pd.read_parquet(features_path)
    feature_list = json.loads(Path(feature_list_path).read_text(encoding="utf-8"))["features"]
    if len(feature_list) != len(set(feature_list)):
        raise ValueError("XGBoost feature list contains duplicate names")
    feature_index = features.set_index("transaction_id")
    missing_ids = set(raw["transaction_id"]).difference(feature_index.index)
    if missing_ids:
        raise ValueError(f"Feature parquet is missing {len(missing_ids)} Phase 1 transactions")
    matrix = feature_index.loc[raw["transaction_id"], feature_list].apply(pd.to_numeric, errors="coerce").to_numpy()
    train_count = int((raw["split"] == "train").sum())
    validation_end = int((raw["split"].isin(["train", "validation"])).sum())
    normalized, normalization = normalize_train_only(matrix, train_count)

    entity_keys: dict[str, list[str | None]] = {}
    entity_maps: dict[str, dict[str, int]] = {}
    non_hub_keys: dict[str, set[str]] = {}
    max_degree: dict[str, int] = {}
    hub_report: dict[str, dict[str, int | float]] = {}
    for entity_type, column in ENTITY_COLUMNS.items():
        keys = [_entity_key(value) for value in raw[column].tolist()]
        unique = list(dict.fromkeys(key for key in keys if key is not None))
        mapping = {key: idx for idx, key in enumerate(unique)}
        entity_keys[entity_type] = keys
        entity_maps[entity_type] = mapping
        train_degrees: dict[str, int] = {}
        for key in keys[:train_count]:
            if key is not None:
                train_degrees[key] = train_degrees.get(key, 0) + 1
        distribution = np.asarray(list(train_degrees.values()), dtype=np.int64)
        cap = max(1, int(np.percentile(distribution, hub_percentile, method="higher"))) if len(distribution) else 1
        max_degree[entity_type] = cap
        full_degrees: dict[str, int] = {}
        for key in keys:
            if key is not None:
                full_degrees[key] = full_degrees.get(key, 0) + 1
        dropped_nodes = {key for key, degree in full_degrees.items() if degree > cap}
        non_hub_keys[entity_type] = set(full_degrees).difference(dropped_nodes)
        dropped_edges = sum(full_degrees[key] for key in dropped_nodes)
        hub_report[entity_type] = {
            "degree_cap_from_train_percentile": cap,
            "train_degree_percentile": hub_percentile,
            "entity_nodes": len(full_degrees),
            "hub_nodes_dropped": len(dropped_nodes),
            "edges_dropped_in_full_snapshot": dropped_edges,
            "remaining_edges_in_full_snapshot": sum(full_degrees.values()) - dropped_edges,
        }

    return GraphBundle(
        transaction_ids=raw["transaction_id"].to_numpy(dtype=np.int64),
        labels=raw["isFraud"].to_numpy(dtype=np.int64),
        splits=raw["split"].to_numpy(),
        feature_names=feature_list,
        transaction_x=torch.from_numpy(normalized),
        entity_keys=entity_keys,
        entity_key_to_index=entity_maps,
        non_hub_keys=non_hub_keys,
        train_count=train_count,
        validation_end=validation_end,
        normalization=normalization,
        hub_report=hub_report,
        max_degree=max_degree,
    )
