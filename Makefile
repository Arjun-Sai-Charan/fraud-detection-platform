SHELL := /bin/bash
UV ?= uv
VENV ?= .venv

.PHONY: setup data features feature-store train train-xgb train-xgb-local train-gnn train-gnn-local drift retrain retrain-local export-model benchmark ring-precision demo-data dashboard k8s-build k8s-up k8s-demo k8s-status k8s-down gnn serve serve-registry test
setup:
	$(UV) venv --python 3.11 $(VENV)
	$(UV) pip install --index-strategy unsafe-best-match --python $(VENV)/bin/python -r requirements.txt

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

drift:
	$(VENV)/bin/python -m src.monitoring.drift

retrain:
	$(VENV)/bin/python -m src.monitoring.retrain --tracking-uri http://localhost:5000 $(RETRAIN_ARGS)

retrain-local:
	$(VENV)/bin/python -m src.monitoring.retrain --tracking-uri file:./mlruns_local $(RETRAIN_ARGS)

gnn: train-gnn

serve:
	$(VENV)/bin/python -m uvicorn src.serving.api:app --host 0.0.0.0 --port 8000

serve-registry:
	FRAUD_MODEL_SOURCE=registry MLFLOW_TRACKING_URI=$(TRACKING_URI) $(VENV)/bin/python -m uvicorn src.serving.api:app --host 0.0.0.0 --port 8000

benchmark:
	$(VENV)/bin/python -m scripts.benchmark_latency

ring-precision:
	$(VENV)/bin/python -m scripts.ring_precision

demo-data:
	$(VENV)/bin/python -m scripts.seed_demo_alerts

dashboard:
	PYTHONPATH=. $(VENV)/bin/streamlit run src/dashboard/app.py --server.port 8501

k8s-build:
	eval "$$(minikube docker-env)" && docker build -f docker/api.Dockerfile -t fraud-api:phase8 . && docker build -f docker/dashboard.Dockerfile -t fraud-dashboard:phase8 . && docker build -f docker/jobs.Dockerfile -t fraud-jobs:phase8 .

k8s-up:
	kubectl apply -f k8s/00-namespace.yaml
	kubectl apply -f k8s/

k8s-demo:
	kubectl delete job producer-demo -n fraud --ignore-not-found
	kubectl apply -f k8s/60-jobs.yaml
	kubectl wait --for=condition=complete job/producer-demo -n fraud --timeout=15m

k8s-status:
	kubectl get pods,services,jobs,cronjobs -n fraud -o wide

k8s-down:
	kubectl delete namespace fraud --ignore-not-found

test:
	$(VENV)/bin/python -m pytest -q

