FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    MPLCONFIGDIR=/tmp/matplotlib
WORKDIR /app

COPY requirements-jobs.txt /app/requirements-jobs.txt
RUN apt-get update && apt-get install --no-install-recommends -y libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir --disable-pip-version-check -r /app/requirements-jobs.txt

RUN groupadd --gid 10001 fraud && useradd --uid 10001 --gid 10001 --create-home fraud \
    && mkdir -p /app/data/processed /app/artifacts/xgb /app/results/drift /app/runtime /mlflow \
    && chown -R 10001:10001 /app /mlflow
COPY --chown=10001:10001 src/common /app/src/common
COPY --chown=10001:10001 src/monitoring /app/src/monitoring
COPY --chown=10001:10001 src/models /app/src/models
COPY --chown=10001:10001 artifacts/xgb /app/artifacts/xgb
COPY --chown=10001:10001 data/processed/features.parquet /app/data/processed/features.parquet
COPY --chown=10001:10001 results/drift/natural_baseline.json /app/results/drift/natural_baseline.json
COPY --chown=10001:10001 results/xgb_metrics.json /app/results/xgb_metrics.json

USER 10001:10001
CMD ["python", "-m", "src.monitoring.drift"]
