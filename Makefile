SHELL := /bin/bash
PYTHON ?= python
VENV ?= .venv

.PHONY: setup data features train gnn serve test
setup:
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/python -m pip install --upgrade pip==25.0.1
	$(VENV)/bin/python -m pip install -r requirements.txt

data:
	$(VENV)/bin/python -m src.data.prepare

features:
	$(VENV)/bin/python -m src.features.build_features

train:
	$(VENV)/bin/python -m src.models.train_xgb

gnn:
	$(VENV)/bin/python -m src.models.train_gnn

serve:
	$(VENV)/bin/python -m uvicorn src.serving.api:app --host 0.0.0.0 --port 8000

test:
	$(VENV)/bin/python -m pytest -q

