from __future__ import annotations

from pathlib import Path
import os
import re
import subprocess
import sys

from packaging.requirements import Requirement
import yaml

ROOT = Path(__file__).resolve().parents[1]


def _requirements(path: Path) -> dict[str, str]:
    pins = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "--")):
            continue
        requirement = Requirement(line)
        assert len(requirement.specifier) == 1, f"Expected one exact pin: {line}"
        specifier = next(iter(requirement.specifier))
        assert specifier.operator == "==", f"Expected exact pin: {line}"
        pins[requirement.name.lower().replace("_", "-")] = specifier.version
    return pins


def _manifests():
    documents = []
    for path in sorted((ROOT / "k8s").glob("*.yaml")):
        documents.extend(document for document in yaml.safe_load_all(path.read_text()) if document)
    return documents


def _pod_specs(document):
    kind = document["kind"]
    if kind in {"Deployment", "Job"}:
        return [document["spec"]["template"]["spec"]]
    if kind == "CronJob":
        return [document["spec"]["jobTemplate"]["spec"]["template"]["spec"]]
    return []


def _memory_mi(value: str) -> int:
    match = re.fullmatch(r"([0-9]+)(Mi|Gi)", value)
    assert match, f"Unexpected memory quantity: {value}"
    amount = int(match.group(1))
    return amount if match.group(2) == "Mi" else amount * 1024


def test_container_requirement_files_use_only_source_pins():
    source = _requirements(ROOT / "requirements.txt")
    for name in ("api", "dashboard", "jobs"):
        selected = _requirements(ROOT / f"requirements-{name}.txt")
        assert selected
        assert set(selected).issubset(source)
        assert all(source[package] == version for package, version in selected.items())

    api = _requirements(ROOT / "requirements-api.txt")
    assert {"fastapi", "uvicorn", "confluent-kafka", "feast", "mlflow", "xgboost"}.issubset(api)
    assert not {"torch", "torch-geometric", "evidently", "streamlit"}.intersection(api)
    assert {"streamlit", "httpx"}.issubset(_requirements(ROOT / "requirements-dashboard.txt"))
    assert {"evidently", "mlflow", "xgboost", "scikit-learn"}.issubset(_requirements(ROOT / "requirements-jobs.txt"))


def test_all_yaml_files_parse():
    paths = list((ROOT / "k8s").glob("*.yaml")) + list((ROOT / "docker").glob("*.yml"))
    assert paths
    for path in paths:
        documents = list(yaml.safe_load_all(path.read_text(encoding="utf-8")))
        assert documents and all(document is not None for document in documents), path


def test_dockerfiles_are_split_pinned_non_root_and_secret_free():
    for image in ("api", "dashboard", "jobs"):
        dockerfile = ROOT / "docker" / f"{image}.Dockerfile"
        text = dockerfile.read_text(encoding="utf-8")
        assert re.search(r"^FROM python:3\.11-slim$", text, re.MULTILINE)
        assert re.search(r"^USER 10001:10001$", text, re.MULTILINE)
        assert "requirements-" + image + ".txt" in text
        assert not re.search(r"(?i)(password|secret|token|api[_-]?key)\s*=\s*\S+", text)
    api_text = (ROOT / "docker/api.Dockerfile").read_text()
    assert "libgomp1" in api_text
    assert "src/feature_repo /app/src/feature_repo" in api_text
    assert "artifacts/xgb /app/artifacts/xgb" in api_text
    assert "artifacts/gnn /app/artifacts/gnn" in api_text
    assert "data/processed/features.parquet" in api_text
    dashboard_text = (ROOT / "docker/dashboard.Dockerfile").read_text()
    assert "results/*.json" in dashboard_text and "results/drift/*.json" in dashboard_text
    jobs_text = (ROOT / "docker/jobs.Dockerfile").read_text()
    assert "libgomp1" in jobs_text
    assert "data/processed/features.parquet" in jobs_text
    assert "artifacts/xgb" in jobs_text


def test_dockerignore_excludes_requested_development_and_sensitive_content():
    ignored = set((ROOT / ".dockerignore").read_text().splitlines())
    assert {".venv", "data/raw", "mlruns_local", "results/*.html", "docs", "notebooks", "tests", "__pycache__"}.issubset(ignored)
    assert ".env" in ignored
    assert "!data/processed/features.parquet" in ignored


def test_manifests_have_resources_non_root_probes_and_local_images():
    docs = _manifests()
    assert {doc["metadata"]["namespace"] for doc in docs if doc["kind"] != "Namespace"} == {"fraud"}
    memory_total = 0
    local_images = {"fraud-api:phase8", "fraud-dashboard:phase8", "fraud-jobs:phase8"}
    observed_local_images = set()
    for document in docs:
        for pod in _pod_specs(document):
            pod_security = pod.get("securityContext", {})
            assert pod_security.get("runAsNonRoot") is True
            for container in pod.get("containers", []):
                resources = container.get("resources", {})
                assert resources.get("requests", {}).get("cpu")
                assert resources.get("requests", {}).get("memory")
                assert resources.get("limits", {}).get("cpu")
                assert resources.get("limits", {}).get("memory")
                memory_total += _memory_mi(resources["limits"]["memory"])
                assert container.get("securityContext", {}).get("runAsNonRoot", pod_security.get("runAsNonRoot")) is True
                image = container["image"]
                if image in local_images:
                    observed_local_images.add(image)
                    assert container["imagePullPolicy"] == "Never"
    assert observed_local_images == local_images
    assert memory_total <= 4291, memory_total  # 4,500 decimal MB expressed in MiB.

    deployments = {doc["metadata"]["name"]: doc for doc in docs if doc["kind"] == "Deployment"}
    assert deployments["kafka"]["spec"]["template"]["spec"]["containers"][0].get("readinessProbe")
    assert deployments["kafka"]["spec"]["template"]["spec"]["containers"][0].get("livenessProbe")
    for name, container_name in (("mlflow", "mlflow"), ("scoring", "api"), ("dashboard", "dashboard")):
        containers = deployments[name]["spec"]["template"]["spec"]["containers"]
        container = next(item for item in containers if item["name"] == container_name)
        assert container.get("readinessProbe") and container.get("livenessProbe")

    scoring_pod = deployments["scoring"]["spec"]["template"]["spec"]
    mounts = {
        container["name"]: {mount["name"]: mount["mountPath"] for mount in container["volumeMounts"]}
        for container in scoring_pod["containers"]
    }
    assert mounts["api"]["serving-runtime"] == mounts["consumer"]["serving-runtime"] == "/runtime"


def test_makefile_build_image_names_match_manifests():
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    build = makefile.split("k8s-build:", 1)[1].split("\nk8s-up:", 1)[0]
    built = set(re.findall(r"-t\s+(fraud-[a-z]+:phase8)", build))
    manifest_images = {
        container["image"]
        for document in _manifests()
        for pod in _pod_specs(document)
        for container in pod.get("containers", [])
        if container["image"].startswith("fraud-")
    }
    assert built == manifest_images == {"fraud-api:phase8", "fraud-dashboard:phase8", "fraud-jobs:phase8"}


def test_compose_fallback_uses_all_three_images_and_shared_runtime_db():
    compose = yaml.safe_load((ROOT / "docker/docker-compose.full.yml").read_text(encoding="utf-8"))
    services = compose["services"]
    assert {"kafka", "mlflow", "api", "consumer", "dashboard"}.issubset(services)
    assert services["api"]["image"] == services["consumer"]["image"] == "fraud-api:phase8"
    assert services["dashboard"]["image"] == "fraud-dashboard:phase8"
    assert services["mlflow"]["image"] == "fraud-jobs:phase8"
    assert services["api"]["volumes"] == services["consumer"]["volumes"]


def test_serving_db_path_can_use_shared_runtime_volume(tmp_path):
    env = os.environ.copy()
    expected = str(tmp_path / "serving.db")
    env["FRAUD_SERVING_DB_PATH"] = expected
    result = subprocess.run(
        [sys.executable, "-c", "from src.common.config import SERVING_DB_PATH; print(SERVING_DB_PATH)"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == expected
