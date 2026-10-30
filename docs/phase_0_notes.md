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

## Phase 0.5 verification status (2026-10-30)

- Working directory: `/home/arjun/fraud-platform`.
- `uname -a`: `Linux ARJUN 6.18.54.1-microsoft-standard-WSL2 #1 SMP PREEMPT_DYNAMIC Tue Oct 6 19:13:07 UTC 2026 x86_64 GNU/Linux`.
- `uv venv --python 3.11 .venv` succeeded and selected CPython 3.11.18.
- `UV_CACHE_DIR=/tmp/fraud-platform-uv-cache make setup` did not complete: uv could not resolve DNS for `download.pytorch.org` while fetching packages. The import check and `pip check` are therefore **UNVERIFIED**; the requirements remain pinned.
- `docker version` reports Docker client 29.9.2, but cannot access `unix:///var/run/docker.sock` (`permission denied`). Compose startup, Kafka round trip, and MLflow endpoint/experiment checks are **UNVERIFIED**.
- Kafka Compose configuration now advertises `kafka:29092` internally and `localhost:9092` externally; this configuration has not been runtime-verified because Docker socket access is blocked.
- Minikube is outside this phase's runtime checks. No data or model metrics are claimed.

To finish the blocked checks from a terminal with package network access and Docker access, run `UV_CACHE_DIR=/tmp/fraud-platform-uv-cache make setup`, then the import and dependency checks in the run instructions; start services with `docker compose -f docker/docker-compose.yml up -d`.

