import pandas as pd

from src.features.categorical import CATEGORICAL_COLUMNS, encode_serving_row, fit_encoder, transform_categories
from src.features.build_features import engineer_causal_features


def _frame(times, amounts=None, devices=None, locations=None):
    n = len(times)
    return pd.DataFrame(
        {
            "TransactionID": range(100, 100 + n),
            "TransactionDT": times,
            "uid": ["same"] * n,
            "TransactionAmt": amounts or [10.0 * (i + 1) for i in range(n)],
            "DeviceInfo": devices or ["device-a"] * n,
            "addr1": locations or [10.0] * n,
        }
    )


def test_first_transaction_edges_and_equal_timestamp_non_leakage():
    frame = _frame([0, 0, 1], [10.0, 20.0, 30.0], ["a", "b", "a"], [1.0, 2.0, 2.0])
    features = engineer_causal_features(frame)
    assert features.loc[0, "is_first_txn_for_uid"] == 1
    assert features.loc[0, "uid_prior_txn_count"] == 0
    assert features.loc[0, "txn_count_last_1h"] == 0
    assert pd.isna(features.loc[0, "amount_to_past_mean"])
    assert pd.isna(features.loc[0, "device_change_vs_previous"])
    assert pd.isna(features.loc[0, "location_change_vs_previous"])
    assert features.loc[1, "uid_prior_txn_count"] == 0
    assert pd.isna(features.loc[1, "amount_to_past_mean"])
    assert features.loc[2, "uid_prior_txn_count"] == 2
    assert features.loc[2, "amount_to_past_mean"] == 2.0
    assert features.loc[2, "device_change_vs_previous"] == 1

    changed_future = frame.copy()
    changed_future.loc[2, "TransactionAmt"] = 99999.0
    changed_features = engineer_causal_features(changed_future)
    pd.testing.assert_series_equal(
        features.loc[:1, "amount_to_past_mean"],
        changed_features.loc[:1, "amount_to_past_mean"],
    )


def test_window_counts_include_boundary_exclude_older_transactions():
    frame = pd.DataFrame(
        {
            "TransactionID": range(8),
            "TransactionDT": [0, 3599, 3600, 3601, 0, 86399, 86400, 86401],
            "uid": ["hour"] * 4 + ["day"] * 4,
            "TransactionAmt": [1.0] * 8,
            "DeviceInfo": ["d"] * 8,
            "addr1": [1.0] * 8,
        }
    )
    features = engineer_causal_features(frame).set_index("transaction_id")
    assert features.loc[1, "txn_count_last_1h"] == 1  # 3599 seconds after prior event.
    assert features.loc[2, "txn_count_last_1h"] == 2  # Exactly 3600 seconds: left boundary included.
    assert features.loc[3, "txn_count_last_1h"] == 2  # The event at 0 is now outside 1h.
    assert features.loc[5, "txn_count_last_24h"] == 1
    assert features.loc[6, "txn_count_last_24h"] == 2  # Exactly 86400 seconds: left boundary included.
    assert features.loc[7, "txn_count_last_24h"] == 2  # The event at 0 is now outside 24h.


def test_uid_hub_flag_uses_prior_history_only():
    frame = _frame(list(range(1002)), [1.0] * 1002)
    features = engineer_causal_features(frame)
    assert features.loc[1000, "uid_prior_txn_count"] == 1000
    assert features.loc[1000, "uid_is_hub"] == 0
    assert features.loc[1001, "uid_prior_txn_count"] == 1001
    assert features.loc[1001, "uid_is_hub"] == 1


def test_saved_categorical_encoder_is_reused_for_serving_and_unknown_values(tmp_path):
    training = pd.DataFrame({column: ["known", None] for column in CATEGORICAL_COLUMNS})
    artifact_path = tmp_path / "encoder.joblib"
    artifact = fit_encoder(training, artifact_path)
    batch = transform_categories(training.iloc[[0]], artifact)
    serving = encode_serving_row({column: "known" for column in CATEGORICAL_COLUMNS}, artifact_path)
    assert serving["ProductCD_encoded"] == batch.iloc[0]["ProductCD_encoded"]
    unknown = encode_serving_row({column: "unseen" for column in CATEGORICAL_COLUMNS}, artifact_path)
    assert all(unknown[f"{column}_encoded"] == -1 for column in CATEGORICAL_COLUMNS)
