# Phase 2: Causal features and Feast

## What this phase does

`src/features/build_features.py` reads the existing Phase 1 train, validation, and test parquet files, combines them in chronological order, and writes `data/processed/features.parquet`. It does not rewrite the Phase 1 split files or EDA outputs. The feature table has the required `transaction_id`, UTC `event_timestamp`, `uid`, features, fraud label, and split columns. The event timestamp is `2017-11-30 00:00:00 UTC + TransactionDT`.

Features are calculated per `uid`. At a transaction timestamp, all rows with that same timestamp are evaluated as one batch: equal-timestamp transactions cannot see each other, and transactions at later timestamps cannot affect earlier rows. The past 1-hour and 24-hour windows include their exact lower boundary and exclude the current event time. Tied rows use `TransactionID` as a stable output ordering; the previous device/location value comes from the last row at the prior distinct timestamp.

## Feature meanings and decisions

- `txn_count_last_1h` and `txn_count_last_24h`: earlier transactions for the same uid in the stated time window. Both are zero when there is no matching history.
- `amount_to_past_mean`: current `TransactionAmt` divided by the mean amount of strictly earlier transactions for this uid. It is null for the first transaction because no past mean exists.
- `device_change_vs_previous` and `location_change_vs_previous`: whether `DeviceInfo` or `addr1` differs from the previous transaction at an earlier timestamp. They are null when there is no previous timestamp.
- `is_first_txn_for_uid`: one for the first observed event for that uid, otherwise zero.
- `uid_prior_txn_count`: the number of strictly earlier transactions for that uid.
- `uid_is_hub`: one after the uid has more than 1,000 strictly earlier transactions; otherwise zero. This keeps the feature causal even for validation and test rows. Hub rows are retained; the flag marks them rather than dropping them.
- Raw fields include `TransactionAmt`, `ProductCD`, `card4`, `card6`, `P_emaildomain`, `R_emaildomain`, `C1`–`C14`, `D1`–`D15`, and `M1`–`M9`. No `V*` fields are included.

Categorical inputs are converted to strings with `__MISSING__` for missing values. A single `OrdinalEncoder` is fit on train rows only and saved as `data/processed/categorical_encoder.joblib`. The same artifact transforms validation/test features and can be loaded with `load_encoder` and `transform_categories` at serving time. Categories absent during training map to `-1`. Both the raw categorical columns and their `_encoded` numeric columns are available in the output.

## Actual local results

The feature table contains **150,000 rows and 72 columns**: 90,000 train, 30,000 validation, and 30,000 test. Its UTC event-time range is **2017-12-01 00:00:00+00:00** to **2018-01-04 19:16:48+00:00**. These values are saved in `results/features_summary.json`.

The Feast repository uses the installed Feast **0.40.1** API. Inspection of that version showed `offline_store: type: file` maps to its Dask file offline store, while `online_store: type: sqlite` accepts the configured local database path. The repo is in `src/feature_repo/`; its file source points at `data/processed/features.parquet`, the entity is `transaction_id`, and the view excludes `isFraud` and `split` so labels and partition metadata are not served as model features.

`feast apply` created the transaction entity and `transaction_features` view. `feast materialize 2017-11-30T00:00:00 2018-01-06T00:00:00` loaded all **150,000** rows into SQLite. The parity script sampled 100 transaction IDs, fetched identical feature refs using `get_historical_features` and `get_online_features`, compared all values (including null handling), and passed: **67 of 67 fields matched**. The actual result is saved in `results/feature_parity.json`.

## Tests

`make test` passed with **8 tests**. Phase 2 tests cover tied timestamps and future perturbations, first-transaction null/zero cases, exact and just-outside 1-hour/24-hour boundaries, causal hub flags, and reuse of the saved categorical encoder including unknown serving values.
