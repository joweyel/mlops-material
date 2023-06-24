# 4. Model Deployment

## 4.1 Three ways of deploying a model

### MLOps: Design, Train, Experiment and Deploy
- **Design**
    - What are requirements for the problem
    - Knowing if ML is the right solution to tackle the problem
- **Train**
    - Experiment tracking (capture metrics from experiments)
    - Productionizing a Jupyter Notebook; turning a Jupyter notebook into a ML pipeline
    - Output: a trained model
- **Operate**
    - Deploy trained model


![deploying models](imgs/deploy.png)

- **`Batch Mode`**
    - Run the model regularly (hourly, daily, monthly, etc.)
    - Get the data from the previous time interval
    - Save the results into database of predictions
    - Often used for marketing related tasks

![scoring-job](imgs/scoring_job.png)

### Batch Mode - Marketin Example

- **Case**: User churning from Taxi to Uber

![example](imgs/marketing_example.png)

### Web Service - Trip Duration Example

![web-service](imgs/web_service.png)
- <u>1-to-1</u> Relationship (Client & Server)
- A user of an taxi-app wants to know the approximate duration of a trip immediately
- Waiting is not viable here and a Webservice has to be ready to go and predict

### Streaming - Taxi Ride
![streaming](imgs/streaming.png)
- <u>1-to-Many</u> Relationship
- Multiple services (consumers) react to streams of events


## 4.2 Web-services: Deploying models with Flask and Docker

Take a look at the Content from the ML-Zoomcamp about deplying models [here](https://github.com/alexeygrigorev/mlbookcamp-code/tree/master/course-zoomcamp/05-deployment).

The content of this section can be found [here](web-service/README.md).

## 4.3 Web-services: Getting the models from the model registry (MLflow)

Initially you have to start an MLflow server with the following command
```bash
mlflow --server --backend-store-uri=sqlite:///mlflow.db --default-artifact-root=s3://<your-s3-bucket>
```

- As model this time a `random forest` model is chosen. The code can be found in [random-forest.ipynb](web-service/random-forest.ipynb).
- Now you have to go to the Mlflow UI and obtain the `run_id` of the trained model. This `run_id` will be used in a `flask` application.
- The next step is to modify the `test.py` and `predict.py` from the `web-service` folder.

- The following code is added to `predict.py` s.t. it can be used with MLflow:
```python
import pickle
import mlflow
from mlflow.tracking import MlflowClient
from flask import Flask, request, jsonify

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
```
- Start [`predict.py`](web-service-mlflow/predict.py) and run [`test.py`](web-service-mlflow/test.py), to see if it returns a value.
- The code abover is relatively convoluted and complex. Therefore it is desired to simplify the process of obtaining the model and other artigacts.
    - We want to define the model and Dictionary Vectorizer as a Pipeline and log the pipeline as artifact instead of `model` and `dv`
    - The pipeline will be stored in the MLflow model-registry
    - See [`random-forest.ipynb`](web-service-mlflow/random-forest.ipynb) for the implementation
    - Changes are made to [`predict.py`](web-service-mlflow/predict.py) in the `predict` function
- There could arise problems when the MLflow Tracking-Server is not running. 
    - To solve this you can directly access the model from an S3 Bucket.
    - It is also advisable to set the `RUN_ID`'s as environment variables. Those can be set accordingly when needed. In the example the variable has to be set in both consoles, where `test.py` and `predict.py` are called.
    ```bash
    export RUN_ID="10f4197008104ad183466cdb19e26c4e"
    ```
### Now you can package the new model to docker (Still under Construction)

- The previoulsy created Dockerfile has to be adapted to work with the new version of web-service. The new Dockefile can be found [here](web-service-mlflow/Dockerfile) (TODO: solving package missmatches).
- To build a container with S3 Access you have to provide the credentials
    - Set `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` as environmental variables
    - Run the build process:
      ```bash
      docker build -t ride-duration-prediction-service:v2 --build-arg AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID --build-arg AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY .
      ```
    - Run the docker container:
    ```bash
    docker run -it --rm -p 9696:9696 ride-duration-prediction-service:v2
    ```




## 4.4 (Optional) Streaming: Deploying models with Kinesis and Lambda 

## 4.5 Batch: Preparing a scoring script

## 4.6 MLOps Zoomcamp 4.6 - Batch: Scheduling batch scoring jobs with Prefect

The unit 4.6 consists of multiple videos:

## 4.7 Choosing the right way of deployment

TODO / COMING SOON

## 4.8 Homework


## Notes

Did you take notes? Add them here:

* [Notes on model deployment (+ creating a modeling package) by Ron M.](https://particle1331.github.io/inefficient-networks/notebooks/mlops/04-deployment/notes.html)
* [Notes on Model Deployment using Google Cloud Platform, by M. Ayoub C.](https://gist.github.com/Qfl3x/de2a9b98a370749a4b17a4c94ef46185)
* [Week4: Notes on Model Deployment by Bhagabat](https://github.com/BPrasad123/MLOps_Zoomcamp/tree/main/Week4)
* [Week 4: Deployment notes by Ayoub.B](https://github.com/ayoub-berdeddouch/mlops-journey/blob/main/deployment-04.md)
* Send a PR, add your notes above this line
