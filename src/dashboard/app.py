"""Streamlit analyst view for stored fraud alerts and offline ring context."""
from __future__ import annotations

import sys
from pathlib import Path

# Streamlit adds the script's directory to sys.path, not necessarily the repo root.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
import streamlit as st

from src.dashboard.data import (
    ROOT,
    fetch_alerts,
    fetch_metrics,
    fetch_rings,
    fetch_transaction,
    last_retrain_decision,
    latest_drift_status,
    read_local_json,
    send_feedback,
)

st.set_page_config(page_title="Fraud analyst dashboard", layout="wide")
st.title("Fraud analyst dashboard")

metrics, metrics_error = fetch_metrics()
alerts_for_kpi, alerts_error = fetch_alerts(limit=1000)
ring_frame, ring_error = fetch_rings()
latency = metrics.get("latency_ms", {})
alert_count = int(metrics.get("alert_count") or 0)
reviewed = int(alerts_for_kpi.get("status", pd.Series(dtype=str)).fillna("open").ne("open").sum()) if not alerts_for_kpi.empty else 0
reviewed_share = reviewed / alert_count if alert_count else 0.0
confirmed = int(alerts_for_kpi.get("status", pd.Series(dtype=str)).eq("confirmed").sum()) if not alerts_for_kpi.empty else 0
confirmed_share = confirmed / reviewed if reviewed else 0.0
drift_text, drift_error = latest_drift_status()
retrain_text, retrain_error = last_retrain_decision()

kpis = st.columns(6)
kpis[0].metric("Alerts", alert_count)
kpis[1].metric("Reviewed share", f"{reviewed_share:.1%}")
kpis[2].metric("Confirmed-fraud share", f"{confirmed_share:.1%}")
kpis[3].metric("Latency p50", "—" if latency.get("p50") is None else f"{latency['p50']:.1f} ms")
kpis[4].metric("Latency p99", "—" if latency.get("p99") is None else f"{latency['p99']:.1f} ms")
kpis[5].metric("Latest drift", drift_text)
if metrics_error:
    st.info(metrics_error)
if alerts_error:
    st.info(alerts_error)
if drift_error:
    st.caption(drift_error)
if retrain_error:
    st.caption(retrain_error)
st.caption(f"Last retrain decision: {retrain_text}")


def render_alerts() -> None:
    statuses = ["All", "open", "under_review", "confirmed", "false_positive"]
    selected_status = st.selectbox("Case status", statuses, key="case_status_filter")
    minimum_score = st.slider("Minimum XGBoost score", 0.0, 1.0, 0.0, 0.01)
    rings = ["All"]
    if not ring_frame.empty:
        rings.extend(sorted(ring_frame["ring_id"].dropna().astype(str).unique().tolist()))
    selected_ring = st.selectbox("Ring", rings, key="ring_filter")
    frame, error = fetch_alerts(status=None if selected_status == "All" else selected_status, limit=1000)
    if error:
        st.info(error)
    if frame.empty:
        st.info("No alerts match the current filters.")
        return
    frame["transaction_id"] = pd.to_numeric(frame["transaction_id"], errors="coerce")
    frame["xgb_score"] = pd.to_numeric(frame["xgb_score"], errors="coerce")
    frame = frame.loc[frame["xgb_score"].ge(minimum_score)].copy()
    ring_members: dict[str, set[int]] = {}
    if not ring_frame.empty:
        for row in ring_frame.to_dict(orient="records"):
            ring_members[str(row["ring_id"])] = {int(value) for value in row.get("transaction_ids", [])}
    if selected_ring != "All":
        frame = frame.loc[frame["transaction_id"].isin(ring_members.get(selected_ring, set()))]
    frame["ring_id"] = frame["transaction_id"].map(
        lambda value: next((ring_id for ring_id, members in ring_members.items() if int(value) in members), None)
    )
    st.dataframe(frame, use_container_width=True, hide_index=True)
    if frame.empty:
        st.info("No alerts match the current filters.")
        return
    transaction_ids = frame["transaction_id"].dropna().astype(int).tolist()
    selected_id = st.selectbox("Case detail", transaction_ids)
    detail, detail_error = fetch_transaction(selected_id)
    if detail_error or detail is None:
        st.info(detail_error or "Case details are unavailable.")
        return
    transaction = detail.get("transaction", {})
    st.subheader(f"Transaction {selected_id}")
    score_cols = st.columns(4)
    score_cols[0].metric("XGBoost", f"{transaction.get('xgb_score', 0):.4f}")
    score_cols[1].metric("GNN", "—" if transaction.get("gnn_score") is None else f"{transaction['gnn_score']:.4f}")
    score_cols[2].metric("Final score", f"{transaction.get('final_score', 0):.4f}")
    score_cols[3].metric("Ring", transaction.get("ring_id") or "None")
    if transaction.get("ring_size"):
        st.caption(f"Ring size: {transaction['ring_size']}")
    contributions = detail.get("top_contributions", [])
    st.markdown("**Top XGBoost feature contributions**")
    if contributions:
        st.dataframe(pd.DataFrame(contributions), use_container_width=True, hide_index=True)
    else:
        st.info("Feature contributions are unavailable.")
    buttons = st.columns(3)
    actions = [
        ("Under review", "pending", "under_review"),
        ("Confirmed fraud", "fraud", "confirmed"),
        ("False positive", "legitimate", "false_positive"),
    ]
    for column, (label, analyst_label, case_status) in zip(buttons, actions):
        if column.button(label, key=f"feedback-{case_status}"):
            result, feedback_error = send_feedback(selected_id, analyst_label, case_status)
            if feedback_error:
                st.error(feedback_error)
            else:
                st.success(f"Saved case status: {result.get('case_status', case_status)}")


alerts_tab, rings_tab = st.tabs(["Alerts", "Rings"])
with alerts_tab:
    auto_refresh = st.toggle("Auto-refresh every 15 seconds", value=False)
    st.fragment(run_every=15 if auto_refresh else None)(render_alerts)()

with rings_tab:
    st.subheader("Linked alert rings")
    if ring_error:
        st.info(ring_error)
    if ring_frame.empty:
        st.info("No offline ring assignments are available.")
    else:
        display_columns = [name for name in ("ring_id", "ring_size", "alert_count", "total_amount", "case_statuses") if name in ring_frame]
        st.dataframe(ring_frame[display_columns], use_container_width=True, hide_index=True)
    precision, precision_error = read_local_json(ROOT / "results/ring_precision.json")
    if precision_error or precision is None:
        st.caption("Offline ring precision comparison is not available yet.")
    else:
        st.markdown("**Offline fraud rate among blend-flagged test transactions**")
        comparison = precision.get("comparison", {})
        ring_cols = st.columns(2)
        inside = comparison.get("inside_multi_transaction_rings", {})
        outside = comparison.get("outside_multi_transaction_rings", {})
        ring_cols[0].metric("Inside multi-transaction rings", f"{inside.get('fraud_rate', 0):.2%}", f"{inside.get('fraud_count', 0)} / {inside.get('transaction_count', 0)} fraud")
        ring_cols[1].metric("Outside rings", f"{outside.get('fraud_rate', 0):.2%}", f"{outside.get('fraud_count', 0)} / {outside.get('transaction_count', 0)} fraud")
        st.caption("This comparison uses ground-truth labels after the fact; those labels are not available during live scoring.")
