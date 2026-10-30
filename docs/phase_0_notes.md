# Phase 0 notes: scaffold

## What was built and how it works

The repository now has the requested folder layout and a shared configuration module. `make setup` creates a Python virtual environment and installs pinned dependencies. Kafka is configured as one combined broker/controller using KRaft, so it has no ZooKeeper service. MLflow uses a SQLite backend and a persistent local artifact directory. Both services have container memory limits.

For a presentation: this phase creates the repeatable foundation for all later data, modeling, streaming, and dashboard work. It does not claim to process data or produce model results yet.

## Key choices

- Python 3.11, CPU-only Torch 2.5.1, and PyG 2.6.1. The official PyG install documentation says PyG 2.3+ can be installed with PyTorch alone; compiled extensions are optional. This avoids architecture-specific extension wheels for the initial project scaffold. Source: [PyG installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).
- Kafka image `apache/kafka:3.9.0`, single node with broker and controller roles.
- Kafka client: pinned `confluent-kafka`.
- Raw input files and generated outputs are gitignored; `.gitkeep` files retain the empty directories.
- The shared threshold value is only a configuration fallback. Training will choose and save the actual alert threshold from validation data.

## How to run

From WSL2 Ubuntu at the repository root:

```bash
make setup
source .venv/bin/activate
python -c "import xgboost, torch, torch_geometric, feast, mlflow, evidently, fastapi, confluent_kafka"
```

## Acceptance checks

- `make setup`: **UNVERIFIED** here. The current execution environment is Windows PowerShell rather than WSL2 Ubuntu, and this check requires network package downloads. Run the commands above in WSL2; successful completion should return exit code 0, with no traceback, and print no output for the import-only command.
- Docker Compose / Kafka / MLflow startup: **UNVERIFIED**. Run `docker compose -f docker/docker-compose.yml up -d`; `docker compose -f docker/docker-compose.yml ps` should show both `kafka` and `mlflow` as running.
- Minikube: outside Phase 0 runtime acceptance; the requested startup command is documented in `docs/windows_setup.md`.
- No data or model metrics exist in this phase, so no result values are claimed.
- Pytest: no Phase 0 behavioral tests exist yet; run `make test` when the virtualenv is installed.

## Known issues to resolve before the next phase

- Validate the Compose Kafka environment against the selected Apache Kafka image and run its health/connectivity check in Docker. Docker is unavailable in the current shell.
- Verify Python dependency resolution and import success from WSL2. Dependencies are pinned but have not been installed in this environment.

