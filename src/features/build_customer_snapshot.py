import boto3
import pandas as pd
from io import BytesIO
from datetime import timedelta


BUCKET = "online-retail-churn-siqi-dev"
PROCESSED_KEY = "processed/online_retail/dt=2026-01-12/transactions.parquet"

LOOKBACK_DAYS = 180
HORIZON_DAYS = 60


def read_parquet_from_s3(bucket: str, key: str) -> pd.DataFrame:
    s3 = boto3.client("s3")
    obj = s3.get_object(Bucket=bucket, Key=key)
    data = obj["Body"].read()
    return pd.read_parquet(BytesIO(data), engine="pyarrow")


def write_parquet_to_s3(df: pd.DataFrame, bucket: str, key: str):
    s3 = boto3.client("s3")
    buf = BytesIO()
    df.to_parquet(buf, index=False, engine="pyarrow")
    buf.seek(0)
    s3.put_object(Bucket=bucket, Key=key, Body=buf.getvalue())


def pick_reference_date(tx: pd.DataFrame, horizon_days: int) -> pd.Timestamp:
    # Ensure we have full future window to build labels
    max_dt = tx["InvoiceDate"].max().normalize()
    return (max_dt - pd.Timedelta(days=horizon_days)).normalize()


def build_snapshot(tx: pd.DataFrame, reference_date: pd.Timestamp) -> pd.DataFrame:
    tx = tx.copy()
    tx["InvoiceDate"] = pd.to_datetime(tx["InvoiceDate"])
    tx["order_date"] = tx["InvoiceDate"].dt.normalize()

    lookback_start = reference_date - pd.Timedelta(days=LOOKBACK_DAYS)
    label_end = reference_date + pd.Timedelta(days=HORIZON_DAYS)

    # 1) Feature window
    feat_tx = tx[(tx["order_date"] >= lookback_start) & (tx["order_date"] <= reference_date)]

    # customer-level aggregates
    agg = (
        feat_tx.groupby("CustomerID")
        .agg(
            last_purchase_date=("order_date", "max"),
            frequency_180d=("InvoiceNo", "nunique"),
            monetary_180d=("line_amount", "sum"),
            days_active_180d=("order_date", "nunique"),
        )
        .reset_index()
    )

    agg["recency_days"] = (reference_date - agg["last_purchase_date"]).dt.days
    agg["aov_180d"] = agg["monetary_180d"] / agg["frequency_180d"]

    # 2) Label window: churn_60d
    future_tx = tx[(tx["order_date"] > reference_date) & (tx["order_date"] <= label_end)]
    future_any = future_tx.groupby("CustomerID").size().rename("future_orders").reset_index()

    snap = agg.merge(future_any, on="CustomerID", how="left")
    snap["future_orders"] = snap["future_orders"].fillna(0).astype(int)
    snap["churn_60d"] = (snap["future_orders"] == 0).astype(int)

    # add metadata columns
    snap["reference_date"] = reference_date
    snap["lookback_days"] = LOOKBACK_DAYS
    snap["horizon_days"] = HORIZON_DAYS

    # tidy
    snap = snap.drop(columns=["future_orders"])
    return snap


if __name__ == "__main__":
    print("Reading processed transactions from S3...")
    tx = read_parquet_from_s3(BUCKET, PROCESSED_KEY)
    print("Tx shape:", tx.shape)

    # sanity: ensure datetime is usable
    tx["InvoiceDate"] = pd.to_datetime(tx["InvoiceDate"])

    reference_date = pick_reference_date(tx, HORIZON_DAYS)
    print("Reference date:", reference_date.date())

    snapshot = build_snapshot(tx, reference_date)
    print("Snapshot shape:", snapshot.shape)
    print(snapshot[["churn_60d", "monetary_180d", "frequency_180d", "recency_days"]].describe())

    out_key = f"features/online_retail/dt=2026-01-12/customer_snapshot_ref={reference_date.date()}.parquet"
    print("Writing snapshot to S3:", out_key)
    write_parquet_to_s3(snapshot, BUCKET, out_key)

    print("Done.")