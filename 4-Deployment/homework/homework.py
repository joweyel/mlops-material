#!/usr/bin/env python
# coding: utf-8

import os
import pickle
import pandas as pd
import argparse

categorical = ["PULocationID", "DOLocationID"]

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df["duration"] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df["duration"] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype("int").astype("str")
    
    return df


def load_model(model_path):
    with open(model_path, "rb") as f_in:
        dv, model = pickle.load(f_in)
    return dv, model


def apply_model(input_file, model_path, output_file, **kwargs):
    year = kwargs["year"]
    month = kwargs["month"]
    
    print(f"Loading data: {input_file}")
    df = read_data(input_file)

    print(f"Loading model: {model_path}")
    dv, model = load_model(model_path)

    print(f"Preprocessing of data")
    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)

    print(f"Applying the model")
    y_pred = model.predict(X_val)
    print(f"Mean prediction duration: {y_pred.mean():.2f}")

    # Creating an artificial `ride_id` column
    df["ride_id"] = f"{year:04d}/{month:02d}_" + df.index.astype("str")

    print(f"Saving the results to {output_file}")
    df_result = pd.DataFrame()
    df_result["predicted_duration"] = y_pred
    df_result["ride_id"] = df["ride_id"]

    # Writing the ride id and the predictions to a dataframe with results
    df_result.to_parquet(
        output_file,
        engine="pyarrow",
        compression=None,
        index=False
    )
    print(f"Saved: {output_file}")


def run():
    parser = argparse.ArgumentParser("Input parameter for the prediction script")
    parser.add_argument("-y", "--year", required=False, default=2023, type=int, help="Year of data")
    parser.add_argument("-m", "--month", required=True, type=int, help="Month of data")
    parser.add_argument("-mo", "--model", required=True, default="model.bin", help="Model path")
    parser.add_argument("-t", "--taxi_type", required=True, default="yellow", help="Color of taxi")
    args = parser.parse_args()

    model_path = args.model
    year = args.year
    month = args.month
    taxi_type = args.taxi_type

    input_file = f"https://d37ci6vzurychx.cloudfront.net/trip-data/{taxi_type}_tripdata_{year:04d}-{month:02d}.parquet"
    output_file = f"output/{taxi_type}/{taxi_type}_tripdata_{year:04d}-{month:02d}.parquet"
    os.makedirs(os.path.split(output_file)[0], exist_ok=True)

    apply_model(
        input_file=input_file,
        model_path=model_path,
        output_file=output_file,
        year=year,
        month=month
    )


if __name__ == "__main__":
    run()