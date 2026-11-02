"""HTTP interface for shared fraud scoring and analyst feedback."""
from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from src.serving.repository import ScoreRepository
from src.serving.scorer import FraudScorer, scorer_from_environment


class ScoreRequest(BaseModel):
    transaction_id: int = Field(ge=1)


class FeedbackRequest(BaseModel):
    transaction_id: int = Field(ge=1)
    analyst_label: str = Field(min_length=1, max_length=64)
    case_status: str = Field(min_length=1, max_length=64)


def create_app(scorer: FraudScorer | Any | None = None, repository: ScoreRepository | None = None) -> FastAPI:
    app = FastAPI(title="Fraud scoring API", version="1.0.0")
    app.state.scorer = scorer if scorer is not None else scorer_from_environment()
    app.state.repository = repository if repository is not None else ScoreRepository()

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

    return app


app = create_app()
