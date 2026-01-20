# Online Retail Churn Targeting (AWS S3 + Batch Scoring)

End-to-end retention targeting project using the UCI Online Retail transaction dataset.
The pipeline ingests raw data into an S3-style lake layout, builds customer snapshots with churn labels, scores customers with a rule-based risk model, applies a segmented incentive policy, and outputs business-ready artifacts (ranked list + run summary).

## What this project delivers
- **scores.parquet**: customer-level scores and policy outputs (risk score, value segment, recommended action/coupon, EV)
- **ranked_list_topK=K.parquet**: the Top-K EV-ranked customer list for campaign execution
- **run_summary.md** (S3): auto-generated run metadata for reproducibility/audit
- **operator_readme.md** (repo): human-readable campaign brief for operators

## Problem framing
We simulate a retention campaign with a capacity limit (Top-K customers).  
For each customer, we estimate:
- **Churn risk proxy** from RFM-style features
- **Customer value proxy** from historical spend
- **Expected Value (EV)** under a segmented coupon policy

EV is computed as:
```EV = churn_risk × uplift × customer_value − incentive_cost```

> Uplift and cost are assumptions (offline simulation). In a real deployment they must be validated via controlled experiments (A/B tests).

## Data windows (single snapshot run)
- Snapshot (reference) date: `2011-10-10`
- Lookback window: 180 days (features)
- Horizon: 60 days (churn label)

Label:
- `churn_60d = 1` if a customer makes **no** purchase in the next 60 days after the reference date.

## Repository structure

- **`artifacts/`**
    - `operator_readme.md`              # Human-facing campaign brief (stable, not auto-generated)
- **`notebooks/`**
    - `01_raw_data_profile.ipynb`      # Raw data profiling (types, missingness, sanity checks)
    - `02_customer_snapshot_exploration.ipynb`
    - `03_train_baseline_model.ipynb`   # Baseline modeling experiments
    - `04_policy_expected_value.ipynb`  # Capacity vs EV curves, policy comparisons
    - `05_artifact_review.ipynb`        # Inspect final artifacts from S3
- **`src/`**
    - `ingest/`
        - `read_raw_from_s3.py`         # Read raw Excel from S3
    - `processing/`
        - `process_raw_to_parquet.py`   # Clean + normalize + write processed parquet to S3
    - `features/`
        - `build_customer_snapshot.py`  # Build customer snapshot (features + churn label)
    - `scoring/`
        - `init.py`
        - `policy.py`                   # Segmented policy assumptions (cost/uplift/coupon)
        - `batch_score.py`            # Batch scoring: scores + ranked list + run summary
- **`.gitignore`**
- **`requirements.txt`**
- **`README.md`**

## S3 layout (data lake style)
This project uses partitioned prefixes to mimic production pipelines:

- `raw/online_retail/dt=YYYY-MM-DD/...`
- `processed/online_retail/dt=YYYY-MM-DD/transactions.parquet`
- `features/online_retail/dt=YYYY-MM-DD/customer_snapshot_ref=YYYY-MM-DD.parquet`
- `scores/online_retail/dt=YYYY-MM-DD/ref=YYYY-MM-DD/scores.parquet`
- `artifacts/online_retail/dt=YYYY-MM-DD/ref=YYYY-MM-DD/ranked_list_topK=K.parquet`
- `artifacts/online_retail/dt=YYYY-MM-DD/ref=YYYY-MM-DD/run_summary.md`

## How to run (local)
### 1) Environment
Create a virtual environment and install dependencies:
```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
Make sure AWS credentials are configured and you can access the S3 bucket (recommended via an IAM user/profile):


### 2) Run pipeline steps

Run each stage in order:

**Ingest / sanity read**
`python src/ingest/read_raw_from_s3.py`

**Raw → processed parquet**
`python src/processing/process_raw_to_parquet.py`

**Build customer snapshot (features + label)**
`python src/features/build_customer_snapshot.py`

**Batch score + write artifacts**
`python -m src.scoring.batch_score`

## Notes
- This is an offline simulation. EV depends on assumed uplift and cost parameters in src/scoring/policy.py.
- To change campaign capacity, update TOPK in batch_score.py and re-run scoring.
- Notebooks are for analysis and review; the pipeline scripts in src/ are the source of truth for reproducible runs.
