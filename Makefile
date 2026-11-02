SHELL := /bin/bash
UV ?= uv
VENV ?= .venv

.PHONY: setup data features feature-store train train-xgb train-xgb-local train-gnn train-gnn-local export-model benchmark gnn serve serve-registry test
setup:
	$(UV) venv --python 3.11 $(VENV)
	$(UV) pip install --python $(VENV)/bin/python -r requirements.txt

data:
	$(VENV)/bin/python -m src.data.prepare

features:
	$(VENV)/bin/python -m src.features.build_features

feature-store:
	$(VENV)/bin/python -m src.features.build_features
	cd src/feature_repo && ../../$(VENV)/bin/feast apply
	cd src/feature_repo && ../../$(VENV)/bin/feast materialize 2017-11-30T00:00:00 2018-01-06T00:00:00
	$(VENV)/bin/python -m src.features.feature_parity

train:
	$(VENV)/bin/python -m src.models.train_xgb

train-xgb:
	$(VENV)/bin/python -m src.models.train_xgb --tracking-uri http://localhost:5000

train-xgb-local:
	$(VENV)/bin/python -m src.models.train_xgb --tracking-uri file:./mlruns_local

TRACKING_URI ?= file:./mlruns_local
export-model:
	$(VENV)/bin/python -m scripts.export_model --tracking-uri $(TRACKING_URI)

train-gnn:
	$(VENV)/bin/python -m src.models.train_gnn --tracking-uri http://localhost:5000

train-gnn-local:
	$(VENV)/bin/python -m src.models.train_gnn --tracking-uri file:./mlruns_local

gnn: train-gnn

serve:
	$(VENV)/bin/python -m uvicorn src.serving.api:app --host 0.0.0.0 --port 8000

serve-registry:
	FRAUD_MODEL_SOURCE=registry MLFLOW_TRACKING_URI=$(TRACKING_URI) $(VENV)/bin/python -m uvicorn src.serving.api:app --host 0.0.0.0 --port 8000

benchmark:
	$(VENV)/bin/python -m scripts.benchmark_latency

test:
	$(VENV)/bin/python -m pytest -q

