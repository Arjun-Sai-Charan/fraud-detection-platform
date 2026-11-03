FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app
WORKDIR /app

COPY requirements-api.txt /app/requirements-api.txt
RUN apt-get update && apt-get install --no-install-recommends -y libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir --disable-pip-version-check -r /app/requirements-api.txt

RUN groupadd --gid 10001 fraud && useradd --uid 10001 --gid 10001 --create-home fraud \
    && mkdir -p /app/data/processed /app/artifacts /runtime \
    && chown -R 10001:10001 /app /runtime
COPY --chown=10001:10001 src/common /app/src/common
COPY --chown=10001:10001 src/serving /app/src/serving
COPY --chown=10001:10001 src/streaming /app/src/streaming
COPY --chown=10001:10001 src/feature_repo /app/src/feature_repo
COPY --chown=10001:10001 artifacts/xgb /app/artifacts/xgb
COPY --chown=10001:10001 artifacts/gnn /app/artifacts/gnn
COPY --chown=10001:10001 data/processed/features.parquet /app/data/processed/features.parquet

USER 10001:10001
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "src.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
