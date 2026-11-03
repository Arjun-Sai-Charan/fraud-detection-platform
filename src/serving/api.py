"""HTTP interface for shared fraud scoring and analyst feedback."""
from __future__ import annotations

from typing import Any
from fastapi import FastAPI, HTTPException, Query
import pandas as pd
from pydantic import BaseModel, Field

from src.common.config import GNN_ARTIFACT_DIR, PROCESSED_DIR
from src.serving.repository import ScoreRepository
from src.serving.scorer import FraudScorer, scorer_from_environment


class ScoreRequest(BaseModel):
    transaction_id: int = Field(ge=1)


class FeedbackRequest(BaseModel):
    transaction_id: int = Field(ge=1)
    analyst_label: str = Field(min_length=1, max_length=64)
    case_status: str = Field(min_length=1, max_length=64)


def create_app(
    scorer: FraudScorer | Any | None = None,
    repository: ScoreRepository | None = None,
    ring_data: tuple[list[dict[str, Any]], dict[int, float]] | None = None,
) -> FastAPI:
    app = FastAPI(title="Fraud scoring API", version="1.0.0")
    app.state.scorer = scorer if scorer is not None else scorer_from_environment()
    app.state.repository = repository if repository is not None else ScoreRepository()
    app.state.ring_data = ring_data

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/score")
    def score(request: ScoreRequest) -> dict[str, Any]:
        try:
            result = app.state.scorer.score(request.transaction_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"No online features for transaction {request.transaction_id}") from exc
        return app.state.repository.save_score(result)

    @app.get("/alerts")
    def alerts(
        limit: int = Query(default=100, ge=1, le=1000),
        status: str | None = Query(default=None, min_length=1, max_length=64),
    ) -> dict[str, list[dict[str, Any]]]:
        return {"alerts": app.state.repository.list_alerts(limit=limit, status=status)}

    @app.post("/feedback")
    def feedback(request: FeedbackRequest) -> dict[str, Any]:
        try:
            return app.state.repository.add_feedback(
                request.transaction_id,
                request.analyst_label,
                request.case_status,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"No score for transaction {request.transaction_id}") from exc

    @app.get("/metrics-lite")
    def metrics_lite() -> dict[str, Any]:
        return app.state.repository.metrics_lite()

    @app.get("/transactions/{transaction_id}")
    def transaction_detail(transaction_id: int) -> dict[str, Any]:
        stored = app.state.repository.get_transaction(transaction_id)
        if stored is None:
            raise HTTPException(status_code=404, detail=f"No stored score for transaction {transaction_id}")
        try:
            contributions = app.state.scorer.feature_contributions(transaction_id, limit=5)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"No online features for transaction {transaction_id}") from exc
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Feature contributions unavailable: {type(exc).__name__}") from exc
        return {"transaction": stored, "top_contributions": contributions}

    @app.get("/rings")
    def rings() -> dict[str, Any]:
        ring_data = app.state.ring_data
        if ring_data is None:
            rings_path = GNN_ARTIFACT_DIR / "rings.parquet"
            features_path = PROCESSED_DIR / "features.parquet"
            if not rings_path.is_file() or not features_path.is_file():
                return {"rings": [], "message": "Offline ring artifacts or processed features are unavailable."}
            try:
                ring_frame = pd.read_parquet(rings_path, columns=["transaction_id", "ring_id", "ring_size"])
                amount_frame = pd.read_parquet(features_path, columns=["transaction_id", "TransactionAmt"])
                ring_data = (
                    ring_frame.where(pd.notna(ring_frame), None).to_dict(orient="records"),
                    dict(zip(amount_frame["transaction_id"].astype(int), amount_frame["TransactionAmt"].fillna(0.0).astype(float))),
                )
                app.state.ring_data = ring_data
            except Exception as exc:
                return {"rings": [], "message": f"Ring data unavailable: {type(exc).__name__}."}
        rows, amounts = ring_data
        return {"rings": app.state.repository.list_rings(rows, amounts), "message": None}

    return app


app = create_app()
