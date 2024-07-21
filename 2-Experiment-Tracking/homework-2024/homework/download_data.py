import os
import sys
import requests

def download_files():
    taxi_type = "green"
    year = 2023
    months = [1, 2, 3]
    url = "https://d37ci6vzurychx.cloudfront.net/trip-data/{taxi_type}_tripdata_{year}-{month:02d}.parquet"

    if not os.path.exists("data"):
        os.makedirs("data", exist_ok=True)

    try:
        for month in months:
            response = requests.get(url.format(taxi_type=taxi_type, year=year, month=month), timeout=4)
            response.raise_for_status()
            with open(f"data/{taxi_type}_tripdata_{year}-{month:02d}.parquet", "wb") as f:
                f.write(response.content)
    except requests.exceptions.RequestException as e:
        print(f"Error downloading with error: {e}")
        sys.exit(1)

    print("Download compled successfully")


if __name__ == "__main__":
    download_files()