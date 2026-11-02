import numpy as np
import pandas as pd

import pytest

from src.models.train_xgb import (
    EXCLUDED_COLUMNS,
    _validate_model_feature_names,
    get_model_features,
    top_budget_mask,
)


def test_model_feature_list_excludes_ids_timestamps_labels_and_split():
    frame = pd.DataFrame(
        {
            "transaction_id": [1, 2],
            "event_timestamp": pd.to_datetime(["2018-01-01", "2018-01-02"], utc=True),
            "uid": ["a", "b"],
            "isFraud": [0, 1],
            "split": ["train", "test"],
            "TransactionAmt": [10.0, 20.0],
            "count_feature": [1, 2],
            "transaction_dt": [123, 456],
        }
    )
    features = get_model_features(frame)
    assert features == ["TransactionAmt", "count_feature"]
    assert set(features).isdisjoint(EXCLUDED_COLUMNS)
    assert "transaction_dt" not in features


def test_model_feature_names_must_be_unique():
    with pytest.raises(ValueError, match="must be unique"):
        _validate_model_feature_names(["TransactionAmt", "TransactionAmt"])


def test_saved_model_features_must_exist_in_feast_online_schema():
    with pytest.raises(ValueError, match="missing from the Feast online schema"):
        _validate_model_feature_names(["TransactionAmt", "missing_feature"], {"TransactionAmt"})


def test_top_budget_threshold_flags_expected_share_and_breaks_score_ties():
    scores = np.array([0.9, 0.9, 0.9, 0.7, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    transaction_ids = np.array([30, 10, 20, 40, 50, 60, 70, 80, 90, 100])
    selected, threshold = top_budget_mask(scores, 0.2, transaction_ids)
    assert selected.sum() == 2
    assert threshold == 0.9
    assert selected[:3].tolist() == [False, True, True]
