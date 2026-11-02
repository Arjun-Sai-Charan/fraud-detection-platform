"""SQLite persistence for served scores, active alerts, and analyst feedback."""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from src.common.config import SERVING_DB_PATH


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ScoreRepository:
    def __init__(self, db_path: Path = SERVING_DB_PATH) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.db_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS scores (
                    transaction_id INTEGER PRIMARY KEY,
                    xgb_score REAL NOT NULL,
                    alert INTEGER NOT NULL,
                    gnn_score REAL,
                    final_score REAL NOT NULL,
                    ring_id TEXT,
                    ring_size INTEGER,
                    latency_ms REAL NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS alerts (
                    transaction_id INTEGER PRIMARY KEY REFERENCES scores(transaction_id) ON DELETE CASCADE,
                    xgb_score REAL NOT NULL,
                    status TEXT NOT NULL DEFAULT 'open',
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    transaction_id INTEGER NOT NULL REFERENCES scores(transaction_id) ON DELETE CASCADE,
                    analyst_label TEXT NOT NULL,
                    case_status TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_alerts_status_created ON alerts(status, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_feedback_transaction ON feedback(transaction_id, created_at DESC);
                """
            )

    def save_score(self, score: dict[str, Any]) -> dict[str, Any]:
        now = _now()
        transaction_id = int(score["transaction_id"])
        alert = bool(score["alert"])
        with self._connect() as connection:
            connection.execute(
                """INSERT INTO scores(transaction_id,xgb_score,alert,gnn_score,final_score,ring_id,ring_size,latency_ms,created_at)
                   VALUES(?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(transaction_id) DO UPDATE SET
                     xgb_score=excluded.xgb_score, alert=excluded.alert, gnn_score=excluded.gnn_score,
                     final_score=excluded.final_score, ring_id=excluded.ring_id, ring_size=excluded.ring_size,
                     latency_ms=excluded.latency_ms, created_at=excluded.created_at""",
                (
                    transaction_id,
                    float(score["xgb_score"]),
                    int(alert),
                    score.get("gnn_score"),
                    float(score["final_score"]),
                    score.get("ring_id"),
                    score.get("ring_size"),
                    float(score["latency_ms"]),
                    now,
                ),
            )
            if alert:
                connection.execute(
                    """INSERT INTO alerts(transaction_id,xgb_score,status,created_at) VALUES(?,?, 'open', ?)
                       ON CONFLICT(transaction_id) DO UPDATE SET xgb_score=excluded.xgb_score""",
                    (transaction_id, float(score["xgb_score"]), now),
                )
            else:
                connection.execute("DELETE FROM alerts WHERE transaction_id=?", (transaction_id,))
        return {**score, "status": "open" if alert else None}

    def add_feedback(self, transaction_id: int, analyst_label: str, case_status: str) -> dict[str, Any]:
        now = _now()
        with self._connect() as connection:
            exists = connection.execute("SELECT 1 FROM scores WHERE transaction_id=?", (int(transaction_id),)).fetchone()
            if not exists:
                raise KeyError(int(transaction_id))
            cursor = connection.execute(
                "INSERT INTO feedback(transaction_id,analyst_label,case_status,created_at) VALUES(?,?,?,?)",
                (int(transaction_id), analyst_label, case_status, now),
            )
            connection.execute(
                "UPDATE alerts SET status=? WHERE transaction_id=?",
                (case_status, int(transaction_id)),
            )
            feedback_id = int(cursor.lastrowid)
        return {
            "id": feedback_id,
            "transaction_id": int(transaction_id),
            "analyst_label": analyst_label,
            "case_status": case_status,
            "created_at": now,
        }

    def list_alerts(self, limit: int = 100, status: str | None = None) -> list[dict[str, Any]]:
        query = "SELECT transaction_id,xgb_score,status,created_at FROM alerts"
        parameters: tuple[Any, ...] = ()
        if status is not None:
            query += " WHERE status=?"
            parameters = (status,)
        query += " ORDER BY created_at DESC, transaction_id DESC LIMIT ?"
        parameters += (int(limit),)
        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [dict(row) for row in rows]

    def metrics_lite(self) -> dict[str, Any]:
        with self._connect() as connection:
            score_count = int(connection.execute("SELECT COUNT(*) FROM scores").fetchone()[0])
            alert_count = int(connection.execute("SELECT COUNT(*) FROM alerts").fetchone()[0])
            feedback_count = int(connection.execute("SELECT COUNT(*) FROM feedback").fetchone()[0])
            latencies = [row[0] for row in connection.execute("SELECT latency_ms FROM scores ORDER BY latency_ms").fetchall()]
        if latencies:
            p50, p99 = np.percentile(np.asarray(latencies, dtype=np.float64), [50, 99]).tolist()
        else:
            p50 = p99 = None
        return {
            "score_count": score_count,
            "alert_count": alert_count,
            "feedback_count": feedback_count,
            "latency_ms": {"p50": p50, "p99": p99},
        }
