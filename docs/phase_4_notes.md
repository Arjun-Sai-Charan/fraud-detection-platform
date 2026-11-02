# Phase 4: Streaming and serving

## What this phase adds

The model export command copies the registered `fraud-xgb` model plus its threshold and feature-list artifacts to `artifacts/xgb/`. That directory is ignored by Git. The API and Kafka consumer share `FraudScorer`: it reads the feature list, asks Feast's SQLite online store for the requested transaction's feature values, applies the saved XGBoost model and threshold, and returns `xgb_score`, `alert`, `gnn_score`, `final_score`, `ring_id`, and `ring_size`. The future GNN and ring fields are present as null values. The per-call latency covers Feast lookup plus model inference.

The scorer defaults to the exported local files. It can instead load the latest registered model from MLflow with `FRAUD_MODEL_SOURCE=registry` and `MLFLOW_TRACKING_URI` set. `ScoreRepository` stores every score, active alerts, and analyst feedback in `data/serving.db`. Feedback is linked to an existing scored transaction and updates the alert's case status.

To serve directly from a registry instead of exporting local files, run `make serve-registry TRACKING_URI=http://localhost:5000`.

The producer replays only the Phase 1 test split, sorted by event timestamp and then transaction ID. `--rate` controls transactions per second and `--limit` caps the replay. The consumer reads `transactions`, calls the same scorer, writes to SQLite, and publishes one response per input transaction to `fraud-scores`. Host commands use `localhost:9092`; `--docker` selects `kafka:29092` for processes on the Compose network.

The Feast view includes `transaction_dt` as time metadata. It is excluded from the XGBoost inputs; the current registered model uses 52 features. Apply and materialize the Feast view before serving so the online store has the current schema.

## Local checks

- The previous 53-feature export and benchmark are superseded. Exported local registered model `fraud-xgb` version 2 to `artifacts/xgb/`; it has 52 inputs and serving threshold `0.1192927360534668`. Loaded it with the default scorer using a mocked Feast response.
- Applied the updated Feast view and materialized 150,000 rows to the local SQLite online store.
- Full test suite: `.venv/bin/python -m pytest -q` completed with **17 passed** (31 dependency deprecation warnings). TestClient/API, SQLite feedback round trip, scorer, Kafka producer/consumer helpers, export helper, producer ordering, and Phase 3's unique/schema feature checks use mocks and temporary local SQLite files.
- After retraining, `scripts/benchmark_latency.py` scored 5,000 chronological test transactions. The measured p50 was **32.1260 ms**, p95 **66.6471 ms**, p99 **86.9035 ms**, throughput **27.2596 transactions/sec**, elapsed **183.4213 sec**. The full values are saved in `results/latency.json`. This measures Feast online lookup plus model inference; it does not measure Kafka.
- **UNVERIFIED: Kafka integration.** The sandbox cannot connect to Kafka at `localhost:9092`; no Kafka broker or network round trip is claimed here.

## Kafka integration commands

Run these from the repository root in your terminal. First export the registered model from your MLflow server and refresh Feast's SQLite online store:

```bash
docker compose -f docker/docker-compose.yml up -d
docker compose -f docker/docker-compose.yml exec kafka /opt/kafka/bin/kafka-topics.sh --bootstrap-server kafka:29092 --create --if-not-exists --topic transactions --partitions 1 --replication-factor 1
docker compose -f docker/docker-compose.yml exec kafka /opt/kafka/bin/kafka-topics.sh --bootstrap-server kafka:29092 --create --if-not-exists --topic fraud-scores --partitions 1 --replication-factor 1
make export-model TRACKING_URI=http://localhost:5000
cd src/feature_repo
../../.venv/bin/feast apply
../../.venv/bin/feast materialize 2017-11-30T00:00:00 2018-01-06T00:00:00
cd ../..
```

Then use separate terminals, all from the repository root:

```bash
make serve
```

```bash
.venv/bin/python -m src.serving.consumer --bootstrap-server localhost:9092
```

```bash
.venv/bin/python -m src.streaming.producer --bootstrap-server localhost:9092 --rate 100 --limit 1000
```

The producer should report `published: 1000`. The consumer should emit 1,000 messages to `fraud-scores` and persist 1,000 scored transactions (assuming a fresh database). The alert count depends on the saved threshold and the replayed transactions; it is the number of those 1,000 score responses whose `alert` value is true. Check the persisted counts with:

```bash
curl http://localhost:8000/metrics-lite
```

Expected fresh-database counts after the consumer finishes: `score_count: 1000`, `alert_count` equal to the number of `alert: true` results, and `feedback_count: 0`. Because the consumer sends one output for every consumed transaction, the `fraud-scores` topic should also receive 1,000 messages.
