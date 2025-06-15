#!/usr/bin/env python

import os
import urllib.request
from argparse import ArgumentParser
import pickle
import pandas as pd


categorical = ['PULocationID', 'DOLocationID']


def load_model(model_path: str):
    """Load model from specified path."""
    with open(model_path, "rb") as f_in:
        dv, model = pickle.load(f_in)
    return dv, model




def read_data(filename):
    df = pd.read_parquet(filename, engine="pyarrow")
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df

def score(input_file, output_path, args):
    """Function for applying the specified model with given data."""
    
    year = args.year
    month = args.month
    taxi_type = args.taxi_type
    model_path = args.model

    print(f"Loading data: {input_file}\n")
    df = read_data(input_file)

    print(f"Loading model: {model_path}\n")
    dv, model = load_model(model_path)
    
    print("Data Pre-Processing", end=" ... ")
    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    print("Done\n")
    
    
    print("Apply model", end=" ... ")
    y_pred = model.predict(X_val)
    print("Done\n")
    
    print(f"Mean prediction ductation: {y_pred.mean():.2f}\n")
    
    # Saving results
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
    
    df_result = pd.DataFrame()
    df_result["pred"] = y_pred
    df_result["ride_id"] = df['ride_id']

    os.makedirs(output_path, exist_ok=True)
    output_file = f"{output_path}/{taxi_type}_scores_{year:04d}-{month:02d}.parquet"
    
    
    df_result.to_parquet(
        output_file,
        engine='pyarrow',
        compression=None,
        index=False
    )
    print(f"Saved: {output_file}")


def __download_data(args):
    taxi_type = args.taxi_type
    year = args.year
    month = args.month
    
    url = f"https://d37ci6vzurychx.cloudfront.net/trip-data/{taxi_type}_tripdata_{year:04d}-{month:02d}.parquet"
    folder_path = f"./data/{args.taxi_type}"
    file_path = f"{folder_path}/{url.split('/')[-1]}"

    os.makedirs(folder_path, exist_ok=True)
    os.system(f"wget -q -c {url} -O {file_path}")
    
    print(f"Local Data obtained: {file_path.split('/')[-1]}")
    return file_path


def download_data(args):
    taxi_type = args.taxi_type
    year = args.year
    month = args.month

    url = f"https://d37ci6vzurychx.cloudfront.net/trip-data/{taxi_type}_tripdata_{year:04d}-{month:02d}.parquet"
    folder_path = f"./data/{taxi_type}"
    file_name = url.split("/")[-1]
    file_path = os.path.join(folder_path, file_name)

    os.makedirs(folder_path, exist_ok=True)

    if not os.path.exists(file_path):  # Avoid re-downloading
        print(f"Downloading from {url}...")
        urllib.request.urlretrieve(url, file_path)
        print(f"Download complete: {file_name}")
    else:
        print(f"File already exists: {file_name}")

    return file_path


def main():
    parser = ArgumentParser("Inputs for scoring script")
    parser.add_argument("-y", "--year", required=False, default=2023, type=int, help="Year of data.")
    parser.add_argument("-m", "--month", required=False, default=3, type=int, help="Month of data.")
    parser.add_argument("-mo", "--model", required=False, default="model.bin", type=str, help="Model path.")
    parser.add_argument("-t", "--taxi_type", required=False, default="yellow", type=str, help="Color of taxi.")
    parser.add_argument("-l", "--local", action="store_true", help="Gets the data and uses it locally.")
    parser.add_argument("-o", "--output", required=False, default="./output", type=str, help="Path where results are saved to.")
    args = parser.parse_args()
    
    
    output_path = f"{args.output}/{args.taxi_type}"
    
    
    if args.local:
        input_file_path = download_data(args)
    else:
        input_file_path: str  = f"https://d37ci6vzurychx.cloudfront.net/trip-data/{args.taxi_type}_tripdata_{args.year:04d}-{args.month:02d}.parquet" 

    print(input_file_path)
    score(input_file_path, output_path, args)
    
    
if __name__ == "__main__":
    main()