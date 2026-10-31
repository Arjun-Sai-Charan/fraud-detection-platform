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

## Phase 0.5 verification status

**VERIFIED by the user in WSL2 (2026-10-31):** the Python 3.11 virtual environment at `.venv` installs cleanly, required imports succeed, `pip check` is clean, the Kafka produce/consume round trip works, and MLflow responds. The user confirmed the pinned versions `pyarrow==18.1.0`, `scikit-learn==1.5.2`, and `sqlalchemy==2.0.36`; these pins are retained unchanged.

