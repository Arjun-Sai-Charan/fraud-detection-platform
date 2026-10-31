# Phase 1: Data preparation and exploratory analysis

## What this phase does

`make data` runs `src.data.prepare` with the Python interpreter in `.venv`. It reads only `train_transaction.csv` and `train_identity.csv`. To keep memory use manageable, it first reads the transaction IDs and timestamps, identifies the earliest 150,000 rows by `TransactionDT`, then reads those rows and matching identity rows in chunks. It left joins on `TransactionID`, sorts chronologically, creates a stable SHA-256 `uid` from `card1`, `card2`, `card3`, `card5`, and `addr1` (missing values become a fixed sentinel), and writes separate parquet files for train, validation, and test.

The split is contiguous and ordered: 60% train, 20% validation, 20% test. No rows are shuffled or randomly sampled. `results/eda_summary.json` is generated from the resulting parquet splits. The executed notebook `notebooks/01_eda.ipynb` contains the tables and plots; its figures are also saved under `results/`.

## Results from the supplied training data

- Rows: **150,000**, with **435 columns** including the generated `uid`.
- `TransactionDT` range: **86,400 to 3,093,408**.
- Overall fraud rate: **2.6467%**. Split rates: train **2.6156%**, validation **1.7200%**, test **3.6667%**. These are contiguous periods, so rates vary over time; the largest split-to-overall difference is about **1.02 percentage points**.
- Split sizes and boundaries:
  - Train: **90,000** rows, `TransactionDT` **86,400–1,884,069**.
  - Validation: **30,000** rows, **1,884,081–2,337,164**.
  - Test: **30,000** rows, **2,337,186–3,093,408**.
- Unique `uid` values: **24,084**. Transactions per uid: minimum **1**, median **2**, mean **6.23**, 90th percentile **10**, 95th percentile **19**, maximum **2,626**.
- A model that always predicts legitimate would score **97.3533% accuracy** on this sample while detecting no fraud. Accuracy alone therefore hides whether the model catches fraud; later evaluation needs fraud recall and precision.

These numbers were computed by the local run and saved in `results/eda_summary.json`; they are not illustrative values.

## EDA figures

The notebook reports top missing-value columns, transaction amount by class, fraud rate by `ProductCD`, `card4`, and `DeviceType`, and daily fraud rate over `TransactionDT`. Figures are saved as `amount_distribution_by_class.png`, `fraud_rate_by_ProductCD.png`, `fraud_rate_by_card4.png`, `fraud_rate_by_DeviceType.png`, `fraud_rate_over_time.png`, and `transactions_per_uid.png` under `results/`.

## Checks

`make data` completed successfully. Focused tests cover strict time ordering between splits, split fraud rates being close to the overall rate on a balanced synthetic fixture, deterministic uids, and a one-row-per-transaction left join. `make test` completed with **4 passed**. No Phase 1 checks required Docker or internet access.
