"""HTTP access and display shaping used by the analyst dashboard."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import httpx
import pandas as pd

API_BASE_URL = os.getenv("FRAUD_API_URL", "http://localhost:8000").rstrip("/")
ROOT = Path(__file__).resolve().parents[2]


def request_json(
    path: str,
    *,
    params: dict[str, Any] | None = None,
    payload: dict[str, Any] | None = None,
    base_url: str | None = None,
    timeout: float = 3.0,
) -> tuple[dict[str, Any] | None, str | None]:
    """Call the API without allowing network or payload errors to escape."""
    try:
        url = f"{(base_url or API_BASE_URL).rstrip('/')}/{path.lstrip('/')}"
        response = httpx.post(url, json=payload, timeout=timeout) if payload is not None else httpx.get(url, params=params, timeout=timeout)
        response.raise_for_status()
        result = response.json()
        if not isinstance(result, dict):
            return None, "The API returned an unexpected response."
        return result, None
    except httpx.HTTPStatusError as exc:
        detail = ""
        try:
            detail = str(exc.response.json().get("detail", ""))
        except (ValueError, AttributeError):
            pass
        return None, detail or f"API request failed with HTTP {exc.response.status_code}."
    except (httpx.HTTPError, ValueError, TypeError):
        return None, "The API is unreachable or returned invalid data. Check that the FastAPI service is running."


def shape_alerts(payload: Any) -> tuple[pd.DataFrame, str | None]:
    try:
        rows = payload.get("alerts", []) if isinstance(payload, dict) else []
        frame = pd.DataFrame(rows)
        if frame.empty:
            return pd.DataFrame(columns=["transaction_id", "xgb_score", "status", "created_at"]), "No alerts are stored yet."
        return frame, None
    except (AttributeError, TypeError, ValueError):
        return pd.DataFrame(columns=["transaction_id", "xgb_score", "status", "created_at"]), "Alert data is unavailable."


def fetch_alerts(
    *, status: str | None = None, limit: int = 100, base_url: str | None = None
) -> tuple[pd.DataFrame, str | None]:
    payload, error = request_json("/alerts", params={"limit": limit, **({"status": status} if status else {})}, base_url=base_url)
    if error:
        return pd.DataFrame(columns=["transaction_id", "xgb_score", "status", "created_at"]), error
    return shape_alerts(payload)


def fetch_metrics(*, base_url: str | None = None) -> tuple[dict[str, Any], str | None]:
    payload, error = request_json("/metrics-lite", base_url=base_url)
    if error or payload is None:
        return {"score_count": 0, "alert_count": 0, "feedback_count": 0, "latency_ms": {"p50": None, "p99": None}}, error
    return payload, None


def fetch_transaction(transaction_id: int, *, base_url: str | None = None) -> tuple[dict[str, Any] | None, str | None]:
    return request_json(f"/transactions/{int(transaction_id)}", base_url=base_url)


def send_feedback(
    transaction_id: int,
    analyst_label: str,
    case_status: str,
    *,
    base_url: str | None = None,
) -> tuple[dict[str, Any] | None, str | None]:
    return request_json(
        "/feedback",
        payload={"transaction_id": int(transaction_id), "analyst_label": analyst_label, "case_status": case_status},
        base_url=base_url,
    )


def shape_rings(payload: Any) -> tuple[pd.DataFrame, str | None]:
    try:
        rows = payload.get("rings", []) if isinstance(payload, dict) else []
        frame = pd.DataFrame(rows)
        message = payload.get("message") if isinstance(payload, dict) else None
        if frame.empty:
            return pd.DataFrame(columns=["ring_id", "ring_size", "alert_count", "total_amount", "case_statuses", "transaction_ids"]), message or "No ring data is available."
        return frame, message
    except (AttributeError, TypeError, ValueError):
        return pd.DataFrame(columns=["ring_id", "ring_size", "alert_count", "total_amount", "case_statuses", "transaction_ids"]), "Ring data is unavailable."


def fetch_rings(*, base_url: str | None = None) -> tuple[pd.DataFrame, str | None]:
    payload, error = request_json("/rings", base_url=base_url)
    if error:
        return shape_rings({"rings": [], "message": error})
    return shape_rings(payload)


def read_local_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
        return (result, None) if isinstance(result, dict) else (None, "The local status file has an unexpected format.")
    except FileNotFoundError:
        return None, "No local status file is available."
    except (OSError, json.JSONDecodeError):
        return None, "The local status file could not be read."


def latest_drift_status(path: Path | None = None) -> tuple[str, str | None]:
    report, error = read_local_json(path or ROOT / "results/drift/recent_window_drift.json")
    if error or report is None:
        return "Unavailable", error
    share = report.get("drift_share")
    threshold = report.get("trigger_threshold")
    if share is None or threshold is None:
        return "Report incomplete", "The drift report is missing its score or threshold."
    try:
        return f"{float(share):.1%} of features; threshold {float(threshold):.1%}", None
    except (TypeError, ValueError):
        return "Report incomplete", "The drift report contains invalid numeric values."


def last_retrain_decision(path: Path | None = None) -> tuple[str, str | None]:
    report, error = read_local_json(path or ROOT / "results/retrain_log.json")
    if error or report is None:
        return "Unavailable", error
    status = str(report.get("status", "unknown"))
    raw_reasons = report.get("trigger_reasons", [])
    reasons = ", ".join(str(reason) for reason in raw_reasons) if isinstance(raw_reasons, list) else ""
    reasons = reasons or "no trigger reason recorded"
    return f"{status}: {reasons}", None
