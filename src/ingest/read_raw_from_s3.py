import boto3
import pandas as pd
from io import BytesIO

BUCKET = "online-retail-churn-siqi-dev"
KEY = "raw/online_retail/dt=2026-01-12/Online Retail.xlsx"

def read_raw_excel_from_s3(bucket: str, key: str) -> pd.DataFrame:
    s3 = boto3.client("s3")
    obj = s3.get_object(Bucket=bucket, Key=key)

    data = obj["Body"].read()          # 读成 bytes
    bio = BytesIO(data)                 # 包装成可 seek 的 file-like
    df = pd.read_excel(bio, engine="openpyxl")
    return df

if __name__ == "__main__":
    df = read_raw_excel_from_s3(BUCKET, KEY)
    print("Shape:", df.shape)
    print(df.head())
    print(df.dtypes)