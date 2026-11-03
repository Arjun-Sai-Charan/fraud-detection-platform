"""Resolve an approved serving model without implicitly serving candidates."""
from __future__ import annotations

from typing import Any

from mlflow.exceptions import MlflowException


def serving_model_version(
    client: Any,
    model_name: str,
    requested_version: str | None = None,
    *,
    serving_alias: str = "champion",
    candidate_alias: str = "candidate",
) -> Any:
    if requested_version:
        return client.get_model_version(model_name, str(requested_version))
    try:
        return client.get_model_version_by_alias(model_name, serving_alias)
    except (MlflowException, AttributeError):
        pass
    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions:
        raise ValueError(f"No registered versions found for model {model_name!r}")
    candidate_version = None
    try:
        candidate_version = str(client.get_model_version_by_alias(model_name, candidate_alias).version)
    except (MlflowException, AttributeError):
        pass
    eligible = [version for version in versions if str(version.version) != candidate_version]
    if not eligible:
        raise ValueError(
            f"No approved serving version for {model_name!r}; candidate alias is not served automatically"
        )
    return max(eligible, key=lambda item: int(item.version))
