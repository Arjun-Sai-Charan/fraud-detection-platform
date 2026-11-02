"""Validation-selected XGBoost/GNN blending and offline ring grouping."""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

ALERT_BUDGETS = (0.01, 0.02, 0.03, 0.05)
BLEND_GRID = tuple(round(value, 1) for value in np.linspace(0.0, 1.0, 11))


def blend_scores(xgb_score: np.ndarray, gnn_score: np.ndarray, weight: float) -> np.ndarray:
    if not 0.0 <= weight <= 1.0:
        raise ValueError("weight must be in [0, 1]")
    xgb_score = np.asarray(xgb_score, dtype=np.float64)
    gnn_score = np.asarray(gnn_score, dtype=np.float64)
    if xgb_score.shape != gnn_score.shape:
        raise ValueError("XGBoost and GNN score arrays must have the same shape")
    return weight * xgb_score + (1.0 - weight) * gnn_score


def choose_blend_weight(
    y_validation: np.ndarray,
    xgb_validation: np.ndarray,
    gnn_validation: np.ndarray,
    grid: tuple[float, ...] = BLEND_GRID,
) -> tuple[float, dict[str, float]]:
    measured = {
        float(weight): float(average_precision_score(
            y_validation, blend_scores(xgb_validation, gnn_validation, float(weight))
        ))
        for weight in grid
    }
    # Stable tie-break: prefer the larger XGBoost weight, reducing GNN influence.
    best = max(grid, key=lambda weight: (measured[float(weight)], float(weight)))
    return float(best), measured


def top_budget_mask(scores: np.ndarray, budget: float, ids: np.ndarray | None = None) -> tuple[np.ndarray, float]:
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1 or not len(scores) or not 0 < budget <= 1:
        raise ValueError("scores must be non-empty and budget must be in (0, 1]")
    ids = np.arange(len(scores)) if ids is None else np.asarray(ids)
    if len(ids) != len(scores):
        raise ValueError("ids and scores must have equal length")
    count = min(len(scores), max(1, math.ceil(len(scores) * budget)))
    ranked = np.lexsort((ids, -scores))[:count]
    mask = np.zeros(len(scores), dtype=bool)
    mask[ranked] = True
    return mask, float(scores[ranked[-1]])


def fit_top3_threshold(scores: np.ndarray, ids: np.ndarray) -> dict[str, Any]:
    mask, threshold = top_budget_mask(scores, 0.03, ids)
    tied = np.asarray(ids)[mask & (np.asarray(scores) == threshold)]
    return {
        "threshold": threshold,
        "alert_budget": 0.03,
        "validation_alerts": int(mask.sum()),
        "validation_rows": int(len(mask)),
        "tie_break": "transaction_id ascending among scores equal to threshold",
        "tie_break_transaction_id_max": int(tied.max()),
    }


def evaluate_scores(y: np.ndarray, scores: np.ndarray, ids: np.ndarray) -> dict[str, Any]:
    y = np.asarray(y, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    positives = int(y.sum())
    budgets = {}
    for budget in ALERT_BUDGETS:
        selected, threshold = top_budget_mask(scores, budget, ids)
        true_positives = int(np.sum(selected & (y == 1)))
        budgets[f"{int(budget * 100)}%"] = {
            "alerts": int(selected.sum()),
            "threshold_score": threshold,
            "precision": float(true_positives / selected.sum()),
            "recall": float(true_positives / positives) if positives else 0.0,
        }
    return {
        "pr_auc": float(average_precision_score(y, scores)),
        "roc_auc": float(roc_auc_score(y, scores)),
        "base_rate": float(y.mean()),
        "alert_budgets": budgets,
    }


def connected_rings(
    transaction_ids: np.ndarray,
    alert_mask: np.ndarray,
    entity_assignments: dict[str, list[str | None]],
    non_hub_keys: dict[str, set[str]],
) -> list[dict[str, Any]]:
    """Assign deterministic connected components to flagged transactions.

    Only components with at least two flagged transactions are called rings;
    other rows receive null ring fields.
    """
    ids = np.asarray(transaction_ids, dtype=np.int64)
    flags = np.asarray(alert_mask, dtype=bool)
    if len(ids) != len(flags):
        raise ValueError("transaction_ids and alert_mask must have equal lengths")
    flagged = np.flatnonzero(flags)
    parent = {int(index): int(index) for index in flagged}
    rank = {int(index): 0 for index in flagged}

    def find(item: int) -> int:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(left: int, right: int) -> None:
        root_left, root_right = find(left), find(right)
        if root_left == root_right:
            return
        if rank[root_left] < rank[root_right]:
            root_left, root_right = root_right, root_left
        parent[root_right] = root_left
        if rank[root_left] == rank[root_right]:
            rank[root_left] += 1

    for entity_type, values in entity_assignments.items():
        owners: dict[str, int] = {}
        allowed = non_hub_keys[entity_type]
        for index in flagged:
            key = values[int(index)]
            if key is None or key not in allowed:
                continue
            if key in owners:
                union(int(index), owners[key])
            else:
                owners[key] = int(index)

    components: dict[int, list[int]] = {}
    for index in flagged:
        components.setdefault(find(int(index)), []).append(int(index))
    ring_for: dict[int, tuple[str, int]] = {}
    for members in components.values():
        if len(members) < 2:
            continue
        ring_id = f"ring-{int(ids[members].min())}"
        for index in members:
            ring_for[index] = (ring_id, len(members))

    return [
        {
            "transaction_id": int(transaction_id),
            "ring_id": ring_for.get(index, (None, None))[0],
            "ring_size": ring_for.get(index, (None, None))[1],
        }
        for index, transaction_id in enumerate(ids)
    ]
