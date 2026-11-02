"""Copy the registered fraud-xgb model and serving metadata to artifacts/xgb."""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

from src.common.config import XGB_ARTIFACT_DIR


def latest_model_version(client: MlflowClient, model_name: str, requested_version: str | None = None):
    if requested_version:
        return client.get_model_version(model_name, str(requested_version))
    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions:
        raise ValueError(f"No registered versions found for model {model_name!r}")
    return max(versions, key=lambda item: int(item.version))


def export_registered_model(
    tracking_uri: str = "file:./mlruns_local",
    model_name: str = "fraud-xgb",
    version: str | None = None,
    output_dir: Path = XGB_ARTIFACT_DIR,
) -> dict[str, str | int]:
    client = MlflowClient(tracking_uri=tracking_uri)
    registered = latest_model_version(client, model_name, version)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="xgb-export-", dir=output_dir.parent) as temp_name:
        staging = Path(temp_name) / "xgb"
        staging.mkdir()
        downloaded_model = Path(client.download_artifacts(registered.run_id, "model", dst_path=temp_name))
        shutil.copytree(downloaded_model, staging / "model")
        for filename, exported_name in (
            ("serving_threshold.json", "threshold.json"),
            ("feature_list.json", "feature_list.json"),
        ):
            source = staging / "model" / filename
            if not source.is_file():
                raise FileNotFoundError(f"Registered model artifact is missing {filename}: {source}")
            shutil.copy2(source, staging / exported_name)
        threshold = json.loads((staging / "threshold.json").read_text(encoding="utf-8"))
        feature_list = json.loads((staging / "feature_list.json").read_text(encoding="utf-8"))
        if not threshold.get("threshold") or not feature_list.get("features"):
            raise ValueError("Registered model threshold or feature list is empty")

        backup = output_dir.with_name(output_dir.name + ".old")
        if backup.exists():
            shutil.rmtree(backup)
        if output_dir.exists():
            output_dir.replace(backup)
        staging.replace(output_dir)
        if backup.exists():
            shutil.rmtree(backup)
    return {
        "model_name": registered.name,
        "version": int(registered.version),
        "output_dir": str(output_dir.resolve()),
        "model_uri": registered.source,
        "feature_count": len(feature_list["features"]),
        "threshold": float(threshold["threshold"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracking-uri", default="file:./mlruns_local")
    parser.add_argument("--model-name", default="fraud-xgb")
    parser.add_argument("--version")
    parser.add_argument("--output-dir", type=Path, default=XGB_ARTIFACT_DIR)
    args = parser.parse_args()
    mlflow.set_tracking_uri(args.tracking_uri)
    result = export_registered_model(args.tracking_uri, args.model_name, args.version, args.output_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
