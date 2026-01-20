import boto3
import pandas as pd
import numpy as np
from io import BytesIO
from datetime import datetime

from src.scoring.policy import POLICY_TABLE, get_policy_params

# ======================
# Config
# ======================
BUCKET = "online-retail-churn-siqi-dev"
DT = "2026-01-12"
REF_DATE = "2011-10-10"
TOPK = 500

SNAPSHOT_KEY = f"features/online_retail/dt={DT}/customer_snapshot_ref={REF_DATE}.parquet"
SCORES_KEY = f"scores/online_retail/dt={DT}/ref={REF_DATE}/scores.parquet"
RANKED_KEY = f"artifacts/online_retail/dt={DT}/ref={REF_DATE}/ranked_list_topK={TOPK}.parquet"
MEMO_KEY = f"artifacts/online_retail/dt={DT}/ref={REF_DATE}/run_summary.md"

# ======================
# IO helpers
# ======================
def read_parquet_from_s3(s3, bucket, key):
    obj = s3.get_object(Bucket=bucket, Key=key)
    return pd.read_parquet(BytesIO(obj["Body"].read()), engine="pyarrow")

def write_parquet_to_s3(s3, df, bucket, key):
    buf = BytesIO()
    df.to_parquet(buf, index=False, engine="pyarrow")
    buf.seek(0)
    s3.put_object(Bucket=bucket, Key=key, Body=buf.getvalue())

def write_text_to_s3(s3, text, bucket, key):
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=text.encode("utf-8"),
        ContentType="text/markdown",
    )

# ======================
# Scoring logic
# ======================
def build_rule_score(df):
    df = df.copy()
    df["recency_rank"] = df["recency_days"].rank(pct=True)
    df["frequency_rank"] = 1 - df["frequency_180d"].rank(pct=True)
    df["monetary_rank"] = 1 - df["monetary_180d"].rank(pct=True)

    df["risk_score"] = (
        0.5 * df["recency_rank"]
        + 0.25 * df["frequency_rank"]
        + 0.25 * df["monetary_rank"]
    )
    return df

def assign_value_segment(df):
    df = df.copy()
    df["value_segment"] = (
        pd.qcut(
            df["monetary_180d"],
            q=3,
            labels=["low", "mid", "high"],
            duplicates="drop",
        )
        .astype("string")
    )
    return df

def apply_policy(df):
    df = df.copy()
    df["p_proxy"] = df["risk_score"]

    def compute_ev(row):
        params = get_policy_params(str(row["value_segment"]))
        return (
            row["p_proxy"]
            * params["uplift"]
            * row["monetary_180d"]
            - params["cost"]
        )

    df["ev"] = df.apply(compute_ev, axis=1)

    df["recommended_coupon_eur"] = (
        df["value_segment"]
        .map(lambda s: get_policy_params(str(s))["coupon_eur"])
        .astype("int64")
    )

    df["recommended_action"] = np.where(
        df["ev"] > 0,
        "send_coupon",
        "no_action",
    )

    return df

# ======================
# Memo
# ======================
def make_run_summary(base_rate, precision_topk, total_ev_topk):
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    policy_lines = "\n".join(
        f"- {k}: coupon €{v['coupon_eur']}, cost €{v['cost']}, uplift {v['uplift']:.0%}"
        for k, v in POLICY_TABLE.items()
    )

    return f"""# Retention Targeting – Run Summary

Generated: {now}

## Run configuration
- dt: {DT}
- reference_date: {REF_DATE}
- topK: {TOPK}

## Policy assumptions (offline simulation)
{policy_lines}

## Offline evaluation metrics
- Base churn rate (all customers): {base_rate:.3f}
- Precision@{TOPK}: {precision_topk:.3f}
- Cumulative Expected Value (Top-{TOPK}): €{total_ev_topk:,.0f}

## Notes
- This document is **auto-generated** for traceability and reproducibility.
- All uplift and cost values are **assumptions**, not observed outcomes.
- Results should be validated via controlled experiments (A/B tests) before rollout.

## Recommended usage
Use the Top-{TOPK} EV-ranked customer list as the candidate pool for the retention campaign.
Apply incentives according to the assigned value segment.
"""

# ======================
# Main
# ======================
if __name__ == "__main__":
    session = boto3.Session(profile_name="siqi-dev")
    s3 = session.client("s3")

    print(f"Reading snapshot from s3://{BUCKET}/{SNAPSHOT_KEY}")
    df = read_parquet_from_s3(s3, BUCKET, SNAPSHOT_KEY)
    print("Snapshot shape:", df.shape)

    df = build_rule_score(df)
    df = assign_value_segment(df)
    df = apply_policy(df)

    scores_cols = [
        "CustomerID",
        "reference_date",
        "churn_60d",
        "recency_days",
        "frequency_180d",
        "monetary_180d",
        "aov_180d",
        "risk_score",
        "value_segment",
        "recommended_action",
        "recommended_coupon_eur",
        "ev",
    ]

    scores = (
        df[scores_cols]
        .sort_values("ev", ascending=False)
        .reset_index(drop=True)
    )

    print(f"Writing scores to s3://{BUCKET}/{SCORES_KEY}")
    write_parquet_to_s3(s3, scores, BUCKET, SCORES_KEY)

    ranked = scores.head(TOPK).copy()
    print(f"Writing ranked list to s3://{BUCKET}/{RANKED_KEY}")
    write_parquet_to_s3(s3, ranked, BUCKET, RANKED_KEY)

    base_rate = float(scores["churn_60d"].mean())
    precision_topk = float(ranked["churn_60d"].mean())
    total_ev_topk = float(ranked["ev"].sum())

    memo = make_run_summary(base_rate, precision_topk, total_ev_topk)
    print(f"Writing memo to s3://{BUCKET}/{MEMO_KEY}")
    write_text_to_s3(s3, memo, BUCKET, MEMO_KEY)

    print("Done.")