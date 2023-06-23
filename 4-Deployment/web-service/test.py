import predict
import requests

# the used features
# For now random values are used
ride = {
    "PULocationID": 10,
    "DOLocationID": 50,
    "trip_distance": 50
}

url = "http://127.0.0.1:9696/predict"
response = requests.post(url, json=ride)
print(response.json())