"""Explicitly promote a registered fraud-xgb candidate to champion."""
from __future__ import annotations

import argparse
import json
from typing import Any

from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

MODEL_NAME = "fraud-xgb"


def promote_candidate(
    version: str,
    *,
    tracking_uri: str = "http://localhost:5000",
    dry_run: bool = False,
    client: Any | None = None,
) -> dict[str, Any]:
    client = client or MlflowClient(tracking_uri=tracking_uri)
    requested = client.get_model_version(MODEL_NAME, str(version))
    try:
        candidate_alias = client.get_model_version_by_alias(MODEL_NAME, "candidate")
        is_candidate = str(candidate_alias.version) == str(version)
    except MlflowException:
        is_candidate = False
    is_candidate = is_candidate or getattr(requested, "tags", {}).get("retraining_role") == "candidate"
    if not is_candidate:
        raise ValueError(f"Model version {version} is not tagged or aliased as a candidate")
    try:
        current = client.get_model_version_by_alias(MODEL_NAME, "champion")
        previous_champion = str(current.version)
    except MlflowException:
        previous_champion = None
    result = {
        "model_name": MODEL_NAME,
        "version": str(version),
        "previous_champion": previous_champion,
        "new_champion": str(version),
        "dry_run": bool(dry_run),
        "promoted": not dry_run,
    }
    if not dry_run:
        client.set_registered_model_alias(MODEL_NAME, "champion", str(version))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True)
    parser.add_argument("--tracking-uri", default="http://localhost:5000")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    result = promote_candidate(
        args.version, tracking_uri=args.tracking_uri, dry_run=args.dry_run
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
