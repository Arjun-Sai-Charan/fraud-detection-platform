# Phase 8: containers and Minikube deployment

## Verification status

**Runtime is UNVERIFIED.** This sandbox did not build Docker images or run Docker Compose, Minikube, or kubectl. YAML parsing, pin comparison, Dockerfile/manifest static checks, image-name checks, non-root/resource/probe checks, and the memory-limit budget were checked offline with the Phase 8 static pytest tests. `kubeconform` is not installed; a Kubernetes schema dry-run was not run here.

All paths are rooted at `/app` in the images. `src.common.config` derives its root from its installed source tree; Feast's `registry` and SQLite online-store paths stay relative to `/app/src/feature_repo`. `FRAUD_SERVING_DB_PATH` selects `/runtime/serving.db`, which is shared by the scoring pod's API and consumer containers. No credentials or secrets are baked into the images.

## Files included and image-size estimates

The API image must contain the already materialized Feast online store. Its SQLite file is **2,046,230,528 bytes (1.91 GiB)**. The SQLite file compressed with local gzip level 1 was **322,113,115 bytes**; this is an observed compression result, not a Docker image measurement. Other relevant copied files are the Feast registry (**2,131 bytes**), processed feature parquet (**14,231,064 bytes / 13.57 MiB**), XGBoost artifacts (**1,196,422 bytes / 1.14 MiB**), and GNN artifacts (**1,035,676 bytes / 0.99 MiB**).

| Image | Exact copied project payload | Installed dependency closure measured in local `.venv` | Estimated uncompressed image size |
|---|---:|---:|---:|
| `fraud-api:phase8` | 2,062,732,803 bytes (1.921 GiB) | 1,446.3 MiB | about 3.45 GiB |
| `fraud-dashboard:phase8` | 388,646 bytes (0.37 MiB) | 340.1 MiB | about 0.45 GiB |
| `fraud-jobs:phase8` | 15,644,457 bytes (14.92 MiB) | 1,556.3 MiB | about 1.65 GiB |

The Python package column is a local installed dependency-closure measurement used as a proxy; the estimates add a roughly 100 MiB slim Python base and project payload. **Final Docker image sizes are UNVERIFIED** until the three images are built. The API image is large because the materialized Feast SQLite store is intentionally baked into it; rebuilding the Feast store is not part of container startup. The API dependency set excludes Torch, PyG, Evidently, and Streamlit.

The processed parquet, model artifacts, materialized Feast `registry.db`/`online.db`, and result JSON files are generated or ignored by Git. Before image builds, reproduce the earlier phases and materialize Feast; Phase 4 model export must provide `artifacts/xgb`, and Phase 5 must provide `artifacts/gnn`. The Docker build intentionally fails if those required inputs are absent rather than producing a partial runtime image.

## Minikube commands

Run from the repository root in WSL2. The machine profile is capped at 10 GiB WSL memory, and this deployment targets five GiB and three CPUs for Minikube.

```bash
minikube start --driver=docker --memory=5120 --cpus=3
make k8s-build
make k8s-up
kubectl wait --for=condition=Ready pod -l app=kafka -n fraud --timeout=5m
kubectl wait --for=condition=Available deployment/mlflow -n fraud --timeout=10m
kubectl wait --for=condition=Available deployment/scoring -n fraud --timeout=10m
kubectl wait --for=condition=Available deployment/dashboard -n fraud --timeout=10m
make k8s-status
```

`make k8s-build` runs `eval "$(minikube docker-env)"` and builds `fraud-api:phase8`, `fraud-dashboard:phase8`, and `fraud-jobs:phase8` inside Minikube's Docker daemon. The manifests use `imagePullPolicy: Never` for these local images.

The static sum of every pod-template memory limit, including the producer Job and drift CronJob, is **4,224 MiB** (about **4.43 decimal GB**), below the 4.5 GB target. This is a configured ceiling, not measured runtime consumption. Kafka is limited to 1,024 MiB, MLflow to 512 MiB, the two scoring containers together to 1,280 MiB, dashboard to 384 MiB, producer Job to 256 MiB, and drift CronJob to 768 MiB.

Start each port-forward in its own terminal and leave it running:

```bash
kubectl port-forward -n fraud svc/scoring 8000:8000
kubectl port-forward -n fraud svc/dashboard 8501:8501
kubectl port-forward -n fraud svc/mlflow 5000:5000
```

Then browse to `http://localhost:8501`; the API health check is `http://localhost:8000/health`, and MLflow is `http://localhost:5000`.

Run the test transaction replay Job (defaults are 1,000 transactions at 20 transactions/second from the ConfigMap):

```bash
make k8s-demo
kubectl get jobs,pods -n fraud -o wide
```

Change the producer defaults by editing the `TRANSACTION_LIMIT` and `PRODUCER_RATE` values in `k8s/10-config.yaml`, then reapply that ConfigMap before starting the Job.

Expected pod shape after deployments become ready (generated pod suffixes and node addresses vary; this is a reference, not captured runtime output):

```text
NAME                         READY   STATUS
kafka-<suffix>               1/1     Running
mlflow-<suffix>              1/1     Running
scoring-<suffix>             2/2     Running
dashboard-<suffix>           1/1     Running
producer-demo-<suffix>       0/1     Completed
```

The producer pod appears after `make k8s-demo`. The drift CronJob is scheduled every six hours; it runs drift reporting followed by `retrain.py --dry-run`, so it does not alter or register models. Kafka, MLflow, and the scoring database use `emptyDir` volumes for this demonstration; their runtime state is lost when pods are recreated. The API and consumer containers share one `emptyDir` in the same scoring pod, avoiding a ReadWriteMany volume. The dashboard's drift/retrain JSON is a build-time snapshot; CronJob outputs are ephemeral and are not reflected in the dashboard. The CronJob also has no shared mount for the scoring database, so it cannot see analyst feedback rows; with no local feedback DB, that trigger count reads as zero. This is a scheduled drift and dry-run check, not a persistent monitoring feedback loop.

Useful logs:

```bash
kubectl logs -n fraud deployment/scoring -c api --tail=100
kubectl logs -n fraud deployment/scoring -c consumer --tail=100
kubectl logs -n fraud deployment/kafka -c broker --tail=100
kubectl logs -n fraud deployment/mlflow -c mlflow --tail=100
kubectl logs -n fraud deployment/dashboard -c dashboard --tail=100
kubectl logs -n fraud job/producer-demo --tail=100
kubectl logs -n fraud job/<drift-check-job-name> --tail=100
```

Remove the namespace and its resources with:

```bash
make k8s-down
```

## Compose fallback

If Minikube is too heavy, build the same three application images in the host Docker daemon (this is separate from `make k8s-build`, which targets Minikube's daemon), then start Compose:

```bash
docker build -f docker/api.Dockerfile -t fraud-api:phase8 .
docker build -f docker/dashboard.Dockerfile -t fraud-dashboard:phase8 .
docker build -f docker/jobs.Dockerfile -t fraud-jobs:phase8 .
docker compose -f docker/docker-compose.full.yml up -d
docker compose -f docker/docker-compose.full.yml ps
```

Compose maps Kafka, MLflow, FastAPI, and Streamlit to host ports 9092, 5000, 8000, and 8501. Kafka advertises `kafka:29092` to containers and `localhost:9092` to the host. API and consumer share the Compose `serving_runtime` volume.

## Troubleshooting

- **ImagePullBackOff:** the locally built image may not be in Minikube's Docker daemon. Re-run `make k8s-build`; check `eval "$(minikube docker-env)" && docker images | grep fraud-`. Confirm the manifest image tag and `imagePullPolicy: Never`.
- **CrashLoopBackOff:** inspect `kubectl describe pod -n fraud <pod>` and `kubectl logs -n fraud <pod> -c <container> --previous`. Confirm the generated inputs were present at build time, the Feast repo contains its registry and online database, and the shared runtime volume is writable by UID 10001.
- **OOMKilled:** inspect the pod events with `kubectl describe pod` and, if metrics-server is available, `kubectl top pods -n fraud`. Keep the Kafka heap below its container limit. The configured limits already total 4,224 MiB; do not raise them past 4.5 decimal GB in aggregate on this profile.
- **Kafka connection refused:** check `kubectl get pods,svc,endpoints -n fraud`, then Kafka broker and consumer logs. In the cluster, clients must use `kafka:29092`; `localhost:9092` is only for host clients in Compose.

## Offline verification run

```bash
.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_phase_8_infra.py
```

This statically parses the Kubernetes and Compose YAML, checks exact dependency pins against `requirements.txt`, verifies image names and Dockerfile non-root settings, checks probes and per-container CPU/memory requests and limits, and asserts that the aggregate Kubernetes pod memory budget stays within 4.5 decimal GB. It does not prove that an image builds or a pod starts. Minikube, Docker Compose, Kubernetes probes, Kafka, MLflow HTTP, and the dashboard runtime are all **UNVERIFIED** here and must be run on the user's machine.
