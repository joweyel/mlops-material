#!/usr/bin/env python
# coding: utf-8

import os
import boto3
import argparse
import uuid # For creating Universally unique identifier
import pandas as pd
import mlflow
from typing import List

def generate_uuid(n: int) -> pd.DataFrame:
    ride_ids = [str(uuid.uuid4()) for _ in range(n)]
    return ride_ids


def read_dataframe(filename: str) -> pd.DataFrame:
    df = pd.read_parquet(filename)

    df["duration"] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df["duration"] = df["duration"].dt.total_seconds() / 60
    df = df[(df["duration"] >= 1) & (df["duration"] <= 60)]
    df["ride_id"] = generate_uuid(len(df))

    return df


def prepare_dictionaries(df: pd.DataFrame) -> List[dict]:
    categorical = ["PULocationID", "DOLocationID"]
    df[categorical] = df[categorical].astype(str)
    df["PU_DO"] = df["PULocationID"] + "_" + df["DOLocationID"]
    categorical = ["PU_DO"]
    numerical = ["trip_distance"]
    dicts = df[categorical + numerical].to_dict(orient="records")
    return dicts


def load_model(run_id):
    logged_model = f"s3://mlflow-artifacts-remote-jw/1/{run_id}/artifacts/model/"
    model = mlflow.pyfunc.load_model(logged_model)
    return model


def apply_model(input_file, run_id, output_file):
    print(f"Reading the data from {input_file}")
    df = read_dataframe(input_file)
    dicts = prepare_dictionaries(df)

    print(f"Loading the model with RUN_ID={run_id}")
    model = load_model(run_id)

    print("Applying the model")
    y_pred = model.predict(dicts)

    print(f"Saving the results to {output_file}")
    df_result = pd.DataFrame()
    df_result["rid_id"] = df["ride_id"]
    df_result["lpep_pickup_datetime"] = df["lpep_pickup_datetime"]
    df_result["PULocationID"] = df["PULocationID"]
    df_result["DOLocationID"] = df["DOLocationID"]
    df_result["actual_duration"] = df["duration"]
    df_result["predicted_duration"] = y_pred
    df_result["diff"] = df_result["actual_duration"] - df_result["predicted_duration"]
    df_result["model_version"] = run_id

    df_result.to_parquet(output_file, index=False)


def run():
    parser = argparse.ArgumentParser("Prediction inputs")
    parser.add_argument("-y", "--year", required=True, type=int, help="Year of data")
    parser.add_argument("-m", "--month", required=True, type=int, help="Month of data")
    parser.add_argument("-t", "--taxi_type", required=True, type=str, help="Type of taxi")
    parser.add_argument("-i", "--run_id", required=True, type=str, help="MLFlow run id")

    args = parser.parse_args()
    year = args.year            # 2021
    month = args.month          # 2
    taxi_type = args.taxi_type  # "green"
    run_id = args.run_id        # "10f4197008104ad183466cdb19e26c4e"
    input_file = f"https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_{year:04d}-{month:02d}.parquet"
    output_file = f"output/{taxi_type}/{taxi_type}_tripdata_{year:04d}-{month:02d}.parquet"
    os.makedirs(os.path.split(output_file)[0], exist_ok=True)

    apply_model(
        input_file=input_file,
        run_id=run_id,
        output_file=output_file
    )

if __name__ == "__main__":
    run()