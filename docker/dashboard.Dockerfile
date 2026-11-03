FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app
WORKDIR /app

COPY requirements-dashboard.txt /app/requirements-dashboard.txt
RUN pip install --no-cache-dir --disable-pip-version-check -r /app/requirements-dashboard.txt

RUN groupadd --gid 10001 fraud && useradd --uid 10001 --gid 10001 --create-home fraud \
    && mkdir -p /app/results/drift && chown -R 10001:10001 /app
COPY --chown=10001:10001 src/dashboard /app/src/dashboard
COPY --chown=10001:10001 results/*.json /app/results/
COPY --chown=10001:10001 results/drift/*.json /app/results/drift/

USER 10001:10001
EXPOSE 8501
CMD ["python", "-m", "streamlit", "run", "src/dashboard/app.py", "--server.address", "0.0.0.0", "--server.port", "8501", "--server.headless", "true"]
