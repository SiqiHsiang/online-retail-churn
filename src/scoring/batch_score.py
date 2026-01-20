import boto3
import pandas as pd
import numpy as np
from io import BytesIO
from datetime import datetime

BUCKET = "online-retail-churn-siqi-dev"
DT = "2026-01-12"
REF_DATE = "2011-10-10"
TOPK = 500

SNAPSHOT_KEY = f"features/online_retail/dt={DT}/customer_snapshot_ref={REF_DATE}.parquet"

SCORES_KEY = f"scores/online_retail/dt={DT}/ref={REF_DATE}/scores.parquet"
RANKED_KEY = f"artifacts/online_retail/dt={DT}/ref={REF_DATE}/ranked_list_topK={TOPK}.parquet"
MEMO_KEY = f"artifacts/online_retail/dt={DT}/ref={REF_DATE}/decision_memo.md"

POLICY_TABLE = {
    "low":  {"cost": 5.0,  "uplift": 0.05, "coupon_eur": 5},
    "mid":  {"cost": 5.0,  "uplift": 0.10, "coupon_eur": 5},
    "high": {"cost": 10.0, "uplift": 0.30, "coupon_eur": 10},
}

def read_parquet_from_s3(s3, bucket, key):
    obj = s3.get_object(Bucket=bucket, Key=key)
    return pd.read_parquet(BytesIO(obj["Body"].read()), engine="pyarrow")

def write_parquet_to_s3(s3, df, bucket, key):
    buf = BytesIO()
    df.to_parquet(buf, index=False, engine="pyarrow")
    buf.seek(0)
    s3.put_object(Bucket=bucket, Key=key, Body=buf.getvalue())

def write_text_to_s3(s3, text, bucket, key):
    s3.put_object(Bucket=bucket, Key=key, Body=text.encode("utf-8"), ContentType="text/markdown")

def build_rule_score(df):
    df = df.copy()
    df["recency_rank"] = df["recency_days"].rank(pct=True)
    df["frequency_rank"] = 1 - df["frequency_180d"].rank(pct=True)
    df["monetary_rank"] = 1 - df["monetary_180d"].rank(pct=True)
    df["risk_score"] = 0.5*df["recency_rank"] + 0.25*df["frequency_rank"] + 0.25*df["monetary_rank"]
    return df

def assign_value_segment(df):
    df = df.copy()
    df["value_segment"] = pd.qcut(df["monetary_180d"], q=3, labels=["low", "mid", "high"], duplicates="drop").astype("string")
    return df

def apply_policy(df):
    df = df.copy()
    df["p_proxy"] = df["risk_score"]

    def ev(row):
        p = POLICY_TABLE[str(row["value_segment"])]
        return row["p_proxy"] * p["uplift"] * row["monetary_180d"] - p["cost"]

    df["ev"] = df.apply(ev, axis=1)
    df["recommended_coupon_eur"] = df["value_segment"].map(lambda s: POLICY_TABLE[str(s)]["coupon_eur"]).astype("int64")
    df["recommended_action"] = np.where(df["ev"] > 0, "send_coupon", "no_action")
    return df

def make_memo(base_rate, precision_topk, total_ev_topk):
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    lines = "\n".join([f"- {k}: coupon €{v['coupon_eur']}, cost €{v['cost']}, uplift {v['uplift']:.0%}"
                       for k, v in POLICY_TABLE.items()])
    return f"""# Retention Targeting Decision Memo

Generated: {now}

## Run config
- dt: {DT}
- reference_date: {REF_DATE}
- topK: {TOPK}

## Policy assumptions (simulation)
{lines}

## Offline sanity metrics
- Base churn rate: {base_rate:.3f}
- Precision@{TOPK}: {precision_topk:.3f}
- Cumulative EV (Top-{TOPK}): €{total_ev_topk:,.0f}

## Recommendation
Use the Top-{TOPK} ranked list for the campaign, applying segmented incentives by value segment.
Validate uplift with an A/B test before scaling.
"""

if __name__ == "__main__":
    # Force profile: never use default chain
    session = boto3.Session(profile_name="siqi-dev")
    s3 = session.client("s3")

    print(f"Reading snapshot from s3://{BUCKET}/{SNAPSHOT_KEY}")
    df = read_parquet_from_s3(s3, BUCKET, SNAPSHOT_KEY)
    print("Snapshot shape:", df.shape)

    df = build_rule_score(df)
    df = assign_value_segment(df)
    df = apply_policy(df)

    scores_cols = [
        "CustomerID","reference_date","churn_60d",
        "recency_days","frequency_180d","monetary_180d","aov_180d",
        "risk_score","value_segment","recommended_action","recommended_coupon_eur","ev"
    ]
    scores = df[scores_cols].sort_values("risk_score", ascending=False).reset_index(drop=True)

    print(f"Writing scores to s3://{BUCKET}/{SCORES_KEY}")
    write_parquet_to_s3(s3, scores, BUCKET, SCORES_KEY)

    ranked = scores.head(TOPK).copy()
    print(f"Writing ranked list to s3://{BUCKET}/{RANKED_KEY}")
    write_parquet_to_s3(s3, ranked, BUCKET, RANKED_KEY)

    base_rate = float(scores["churn_60d"].mean())
    precision_topk = float(ranked["churn_60d"].mean())
    total_ev_topk = float(ranked["ev"].sum())

    memo = make_memo(base_rate, precision_topk, total_ev_topk)
    print(f"Writing memo to s3://{BUCKET}/{MEMO_KEY}")
    write_text_to_s3(s3, memo, BUCKET, MEMO_KEY)

    print("Done.")