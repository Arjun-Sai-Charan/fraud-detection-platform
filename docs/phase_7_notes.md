# Phase 7: analyst dashboard

## What it does

The dashboard reads stored alerts from the FastAPI service, lets an analyst filter by status, score, and offline ring, and opens a case with its stored XGBoost/GNN/blended scores and the five largest signed XGBoost feature contributions. The contribution endpoint fetches that transaction's features from the Feast online store and asks the XGBoost booster for `pred_contribs`. Case buttons submit feedback to the existing `/feedback` endpoint.

The new `/rings` endpoint summarizes the offline Phase 5 ring assignments with member count, stored alert count and statuses, and total transaction amount. The dashboard also shows the recent-window drift report and last retraining decision from local JSON files. Existing API endpoints and their response schemas were left unchanged.

The ring precision calculation uses the test labels only for an offline analysis. The saved validation top-3% blend threshold flagged **1,184 / 30,000** test transactions. Of these, **893** were in multi-transaction rings, with fraud rate **49.05%** (438 fraud); **291** were outside rings, with fraud rate **59.11%** (172 fraud). Rings did not improve the fraud rate in this comparison. The label is ground truth available only after the fact, never at live scoring time. Results are in [ring_precision.json](../results/ring_precision.json).

Across **46** assigned rings, median size was **2**, p90 **9.5**, and maximum **727**; the largest ring contains **61.4%** of all flagged test rows. Strong-only links had a **62.14%** fraud rate (87/140) versus **59.11%** outside rings, a modest difference; rings including weak links were **46.61%** (351/753), so these results do not justify a separate strong-rings dashboard filter.

`make demo-data` scores a chronological, evenly spaced sample of up to 300 test transactions with the same local scorer and stores them in `data/serving.db` without Kafka. The local seed run scored **300** rows and saved **15** alerts.

## Run the demo

Run these commands from the repository root in WSL2 after the Feast repository and online store have already been materialized, and after Phase 4's local XGBoost artifacts are present. The dashboard demo does not rebuild the feature store.

```bash
make serve
```

Keep the API running in the first terminal. In a second terminal, seed the local SQLite database; this step does not require Kafka:

```bash
rm -f data/serving.db
make demo-data
```

In a third terminal, launch Streamlit and open `http://localhost:8501`:

```bash
make dashboard
```

If the API runs at a different address, set the dashboard base URL explicitly:

```bash
FRAUD_API_URL=http://localhost:8000 make dashboard
```

To regenerate the offline ring comparison, run:

```bash
make ring-precision
```

For screenshots, capture (1) the KPI row and filtered Alerts table, (2) a selected case showing scores/contributions and feedback buttons, and (3) the Rings tab with its offline label caveat. Save the images under `docs/screenshots/` and include them in the project review.

## Verification limits

The full local pytest suite passed (**42 tests**). It covers the API additions with temporary SQLite, HTTP data shaping, empty/unreachable API behavior, ring calculation, scorer contribution method, and the Streamlit `AppTest` smoke test. The demo seed also ran against the local model artifacts and Feast store. Live verification of the FastAPI service at `http://localhost:8000` and the browser dashboard is **UNVERIFIED** until run in the user's terminal. The rings are an offline artifact, so they may lag later streamed transactions.
