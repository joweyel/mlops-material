import os
from urllib.request import urlretrieve
import pickle
import mlflow.sklearn
import mlflow.sklearn
import pandas as pd
import numpy as np
import scipy
import sklearn
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
import mlflow
from prefect import flow, task
from typing import Tuple


@task(retries=2, retry_delay_seconds=10)
def download_to_path(data_path: str) -> None:
    """Obtains the data when not already locally available."""
    if os.path.exists(data_path):
        print(f"Already downloaded: {data_path.split('/')[-1]}")
        return
    file_name: str = data_path.split("/")[-1]
    url: str = f"https://d37ci6vzurychx.cloudfront.net/trip-data/{file_name}"

    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    print(f"Downloading {file_name}")
    urlretrieve(url, data_path)
    print(f"Downloaded {file_name} to {data_path}")


@task(retries=3, retry_delay_seconds=2)
def read_dataframe(filename) -> pd.DataFrame:
    df = pd.read_parquet(filename, engine="pyarrow")
    print("Records(Q3): ", df.shape[0])

    df["duration"] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df.duration = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ["PULocationID", "DOLocationID"]
    df[categorical] = df[categorical].astype(str)
    print("Records(processed): ", df.shape[0])

    return df


@task
def train_model(df: pd.DataFrame) -> Tuple[
    pd.DataFrame,
    np.ndarray,
    sklearn.feature_extraction.DictVectorizer,
]:

    features = ["PULocationID", "DOLocationID"]
    df[features] = df[features].fillna(int(0))
    df[features] = df[features].astype("str")
    df_dicts = df[features].to_dict(orient="records")

    dv = DictVectorizer()
    X = dv.fit_transform(df_dicts)
    y = df["duration"].values

    mlflow.sklearn.autolog()

    with mlflow.start_run():
        model = LinearRegression()
        model.fit(X, y)
        print(f"intercept_(Q5): {model.intercept_:.2f}")

        # Log Preprocessor
        os.makedirs("models", exist_ok=True)
        prep_path = "models/preprocessor.b"
        with open(prep_path, "wb") as f_dv:
            pickle.dump(dv, f_dv)
        mlflow.log_artifact(prep_path, artifact_path="preprocessor")


@flow(name="main-flow-hw3")
def main_flow_hw3(data_path: str = "./data/yellow_tripdata_2023-03.parquet") -> None:

    # MLflow settings
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("nyc-taxi-experiment-hw3")

    # Obtain Data
    download_to_path(data_path)

    # Load Data
    df = read_dataframe(data_path)
    print("Records(Q4): ", df.shape[0])

    # Train Model
    train_model(df)


if __name__ == "__main__":
    data_path: str = "./data/yellow_tripdata_2023-03.parquet"
    main_flow_hw3(data_path)
