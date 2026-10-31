import pandas as pd
import pytest

from src.data.prepare import build_uid, load_and_join, split_ordered


def _frame(n=100):
    return pd.DataFrame(
        {
            "TransactionID": range(1, n + 1),
            "TransactionDT": range(1000, 1000 + n),
            "isFraud": [int(i % 10 == 0) for i in range(n)],
            "TransactionAmt": [float(i + 1) for i in range(n)],
            "card1": [i % 7 for i in range(n)],
            "card2": [None if i % 4 == 0 else i % 5 for i in range(n)],
            "card3": [150] * n,
            "card5": [None] * n,
            "addr1": [i % 3 for i in range(n)],
        }
    )


def test_splits_are_chronological_and_fraud_rates_track_overall():
    splits = split_ordered(_frame())
    train, validation, test = (splits[name] for name in ("train", "validation", "test"))
    assert train.TransactionDT.max() < validation.TransactionDT.min()
    assert validation.TransactionDT.max() < test.TransactionDT.min()
    overall = pd.concat(splits.values()).isFraud.mean()
    for part in splits.values():
        assert abs(part.isFraud.mean() - overall) < 0.02


def test_uid_is_deterministic_and_uses_missing_sentinel():
    frame = _frame(3)
    assert build_uid(frame).equals(build_uid(frame.copy()))
    changed_missing_card = frame.copy()
    changed_missing_card.loc[0, "card2"] = None
    assert build_uid(frame).iloc[0] == build_uid(changed_missing_card).iloc[0]


def test_training_join_is_left_and_one_row_per_transaction(tmp_path):
    transaction_path = tmp_path / "train_transaction.csv"
    identity_path = tmp_path / "train_identity.csv"
    frame = _frame(12)
    frame.to_csv(transaction_path, index=False)
    pd.DataFrame({"TransactionID": [2, 4, 6], "DeviceType": ["a", "b", "c"]}).to_csv(identity_path, index=False)
    joined = load_and_join(transaction_path, identity_path)
    assert len(joined) == len(frame)
    assert joined.TransactionID.is_unique
    assert joined.DeviceType.notna().sum() == 3
    assert joined.TransactionDT.is_monotonic_increasing


def test_duplicate_identity_transaction_is_rejected(tmp_path):
    transactions = _frame(3)
    tp, ip = tmp_path / "tx.csv", tmp_path / "id.csv"
    transactions.to_csv(tp, index=False)
    pd.DataFrame({"TransactionID": [1, 1], "DeviceType": ["a", "b"]}).to_csv(ip, index=False)
    with pytest.raises(ValueError, match="duplicate TransactionID"):
        load_and_join(tp, ip)
