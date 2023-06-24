import os
import pickle
import mlflow
from mlflow.tracking import MlflowClient
from flask import Flask, request, jsonify


""" # Old Version (model & dv read in seperately)
# Get connected!
RUN_ID = "d0d81ceebeac478cbd2f2b5ab20ec22f"
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

# Download and load the DictVectorizer
path = client.download_artifacts(run_id=RUN_ID, path="dict_vectorizer.bin")
print(f"Downloading the dict vectorizer to {path}")
with open(path, "rb") as f_in:
    dv = pickle.load(f_in)

# Load the trained sklearn-model
logged_model = f'runs:/{RUN_ID}/model'
model = mlflow.pyfunc.load_model(logged_model)
"""

MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
# Set export RUN_ID="10f4197008104ad183466cdb19e26c4e"
RUN_ID = os.getenv("RUN_ID")  # with pipeline as artifact
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Old Version v2: Not usable when tracking-server is down!
# logged_model = f'runs:/{RUN_ID}/model'

logged_model = f"s3://mlflow-artifacts-remote-jw/1/{RUN_ID}/artifacts/model"
model = mlflow.pyfunc.load_model(logged_model) 

def prepare_features(ride):
    features = {}
    features['PU_DO'] = '%s_%s' % (ride["PULocationID"], ride["DOLocationID"])
    features['trip_distance'] = ride['trip_distance']
    return features

def predict(features):
    preds = model.predict(features)
    return float(preds[0])

# Creating a flask application
app = Flask("duration-prediction")

# Converting the endpoint function to an HTTP endpoint
@app.route("/predict", methods=["POST"])
def predict_endpoint():
    ride = request.get_json()
    features = prepare_features(ride)
    pred = predict(features)

    result = {
        "duration": pred, 
        "model_version": RUN_ID
    }
    return jsonify(result)

if __name__ == "__main__":
    # For development purposes locally
    app.run(debug=True, host="0.0.0.0", port=9696)