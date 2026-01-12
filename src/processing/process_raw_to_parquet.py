import boto3
import pandas as pd
from io import BytesIO

BUCKET = "online-retail-churn-siqi-dev"

RAW_KEY = "raw/online_retail/dt=2026-01-12/Online Retail.xlsx"
PROCESSED_KEY = "processed/online_retail/dt=2026-01-12/transactions.parquet"

def force_string_series(s: pd.Series) -> pd.Series:
    return s.map(lambda x: pd.NA if pd.isna(x) else str(x)).astype("string")

def read_raw_excel_from_s3(bucket: str, key: str) -> pd.DataFrame:
    s3 = boto3.client("s3")
    obj = s3.get_object(Bucket=bucket, Key=key)

    data = obj["Body"].read()
    df = pd.read_excel(BytesIO(data), engine="openpyxl")
    return df

def normalize_schema(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["InvoiceNo"] = force_string_series(df["InvoiceNo"])
    df["StockCode"] = force_string_series(df["StockCode"])
    df["Description"] = df["Description"].astype("string")
    df["Country"] = df["Country"].astype("string")
    df["CustomerID"] = (
        pd.to_numeric(df["CustomerID"], errors="coerce")
        .astype("Int64")
    )

    return df

def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_schema(df)

    df = df[~df["InvoiceNo"].str.startswith("C", na=False)]
    df = df[df["Quantity"] > 0]
    df = df[df["UnitPrice"] > 0]
    df = df.dropna(subset=["CustomerID"])

    df["line_amount"] = df["Quantity"] * df["UnitPrice"]

    return df


def write_parquet_to_s3(df: pd.DataFrame, bucket: str, key: str):
    s3 = boto3.client("s3")

    buffer = BytesIO()
    df.to_parquet(buffer, index=False, engine="pyarrow")
    buffer.seek(0)

    s3.put_object(Bucket=bucket, Key=key, Body=buffer.getvalue())


if __name__ == "__main__":
    print("Reading raw data from S3...")
    raw_df = read_raw_excel_from_s3(BUCKET, RAW_KEY)
    print(f"Raw shape: {raw_df.shape}")

    print("Cleaning transactions...")
    processed_df = clean_transactions(raw_df)
    print(f"Processed shape: {processed_df.shape}")

    print(processed_df[["InvoiceNo","StockCode","CustomerID"]].dtypes)
    print(processed_df["StockCode"].map(type).value_counts().head())

    print("Writing processed parquet to S3...")
    write_parquet_to_s3(processed_df, BUCKET, PROCESSED_KEY)

    print("Done.")