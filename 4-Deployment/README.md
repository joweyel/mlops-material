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

**Types of Deployments: Overview**

![deploying models](imgs/deploy.png)

- **`Batch Mode`**
    - Run the model regularly (hourly, daily, monthly, etc.)
    - Get the data from the previous time interval (not real-time)
    - Save the results into database of predictions
    - Often used for marketing related tasks

![scoring-job](imgs/scoring_job.png)

### Batch Mode - Marketing Example

- **Case**: User churning from Taxi to Uber

![example](imgs/marketing_example.png)

### Web Service (1-to-1) - Trip Duration Example

![web-service](imgs/web_service.png)
- <u>1-to-1</u> Relationship (Client and Server)
- A user of an taxi-app wants to know the approximate duration of a trip immediately
- Waiting is not viable here and a web service has to be ready to go and predict
- More or less real time

### Streaming (1-to-N) - Taxi Ride
![streaming](imgs/streaming.png)
- <u>1-to-Many</u> Relationship (Producer and Consumer)
- Multiple services (consumers) react to streams of events
- The consumers can do predictions for specific tasks (based on the incoming event) and return it to the producer of the data-input


## 4.2 Web-services: Deploying models with Flask and Docker

Take a look at the content from the ML-Zoomcamp about deplying models [here](https://github.com/alexeygrigorev/mlbookcamp-code/tree/master/course-zoomcamp/05-deployment) for a more comprehensive overview over the topic.

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
    - To solve this you can directly access the model from an *S3* Bucket.
    - It is also advisable to set the `RUN_ID`'s as environment variables. Those can be set accordingly when needed. In the example the variable has to be set in both consoles, where `test.py` and `predict.py` are called.
    ```bash
    export RUN_ID="10f4197008104ad183466cdb19e26c4e"
    ```
<!-- ### Now you can package the new model to docker (Still under Construction)

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
    ``` -->

## 4.4 Streaming: Deploying models with Kinesis and Lambda

**Machine Learning for Streaming**
- Scenario
- Creating the role
- Create a `Lambda` Function and test it
- Create a `Kinesis` stream
- Connect the function to the stream
- Send the records

**Links:**
- [Tutorial: Using Amazon Lambda with Amazon Kinesis](https://docs.aws.amazon.com/lambda/latest/dg/with-kinesis.html)

### Granting Access and defining `AWSLambdaKinesisExecutionRole`
To work with AWS Lambda and Kinesis, it is required to attach a specific role to the (IAM) user that utilizes the services. You also have to have access to the AWS Lambda and Kinesis services already granted.
- Go to `IAM` in the AWS web-ui and choose `Access management` $\rightarrow$ `Roles` and `Create role`
- **Step 1: Select trusted entity** 
  - As `Trusted entity type` select `AWS service` and as `Use case` select `Lambda`
- **Step 2: Add permissions**
  - In `Permissions policies` select `AWSLambdaKinesisExecutionRole`
- **Step 3: Name, review, and create**
  - As `Role name` use `lambda-kinesis-role`
  - Click `Create role` and your done!

### Creating the AWS Lambda function
1. Go to `Lambda` in the AWS web-ui and click `Create functions`
   - Select `Author from scratch` 
   - **Function name**: `ride-duration-prediction-test`
   - **Runtime**: `Python 3.9`
   - **Architecture**: `x86_64`
   - `Change default execution role` should be `Use an existing role` (`lambda-kinesis-role`)
  
### Creating `Kinesis` Datastream and connecting to it
2. Go to `Kinesis` on the AWS web-ui and click `Create data stream`
   - Relevant documentation: [Create a Kinesis stream](https://docs.aws.amazon.com/en_us/lambda/latest/dg/with-kinesis-example.html)
   - Name it `ride_events`, choose `Provisioned` mode and set `Provisioned shards` to 1 (can vary depending on anticipated traffic)
   - Click `Create data stream` and the stream will be created

### Connect (Lambda) to Kinesis
3. Add a trigger to your Lambda function and connect the previously created Kinesis stream to it
   - For **Kinesis stream** choose `kinesis/ride_events`
   - The rest can stay the same
  
### Send a test event to the stream
For this see the follwoing [README](streaming/README.md)

### Create new Lambda from Container Image
In this section, the a Lambda function is connected to the docker container that was prevoiously pushed to AWS ECR

1. Create new Lambda function
   - Choose `Container Image`
   - **Function name**: `ride-durection-prediction`
   - **Container image URI**: Browse and select `duration-model` or a URI like this `886638369043.dkr.ecr.us-east-1.amazonaws.com/duration-model:v1`
   - In **Change default execution role**
     - Choose `Use an existing roll` and select `lambda-kinesis-role`

2. Add environment variables at `[Configuration] -> Environment variables`
   - **PREDICTIONS_STREAM_NAME**: `ride_predictions`
   - **RUN_ID**: `10f4197008104ad183466cdb19e26c4e` (change to your run id)
3. Add trigger to Kinesis stream
   - Select Kinesis and then the stream `ride_events`

4. Delete the old Lambda function that was used for testing

5. Test the Lambda function with the following inputs:
   ```bash
   KINESIS_STREAM_INPUT=ride_events
   aws kinesis put-record \
    --stream-name ${KINESIS_STREAM_INPUT} \
    --partition-key 1 \
    --cli-binary-format raw-in-base64-out \
    --data '{
        "ride": {
            "PULocationID": 130,
            "DOLocationID": 205,
            "trip_distance": 3.66
        }, 
        "ride_id": 156
    }'
   ```
   - There is still a problem of Lambda not having access to S3 ressources
6. Add permissions for S3 access to `lambda-kinesis-role` so that the Lambda function can access the data
   - Go to `IAM`, then `Roles` and search for `lambda-kinesis-role`
   - Select `Add permissions` and choose `Create inline policy`
   - Select `List` and `Read` permisiions
   - In the `Resources` menu click on `Add ARNs` in the **bucket** part
     - Provide the path to the bucket 
   - In the `Resources` menu click on `Add ARNs` in the **object** part
     - Provide the path to the bucket in `Resource bucket name`
     - In `Resource object name` select `Any object name`
   - With a little bit of shortening (while retaining the functionality) you will get something like this:
     ```json
     {
        "Version": "2012-10-17",
        "Statement": [
          {
            "Sid": "VisualEditor0",
            "Effect": "Allow",
            "Action": [
              "s3:Get*",
              "s3:List*"
            ],
            "Resource": [
              "arn:aws:s3:::mlflow-artifacts-remote-jw ",
              "arn:aws:s3:::mlflow-artifacts-remote-jw/*"
            ]
          }
        ]
     }
     ```
   - As policy name use (Exemplary! Replace with your S3 bucket name): `read_permission_mlflow-artifacts-remote-jw`
   - Click `Create policy` and you are done!
7. Testing the Lambda Function
   - Create a Test with the name of `test` and provide the following input:
     ```json
     {
        "Records": [
            {
                "kinesis": {
                    "kinesisSchemaVersion": "1.0",
                    "partitionKey": "1",
                    "sequenceNumber": "49630081666084879290581185630324770398608704880802529282",
                    "data": "ewogICAgICAgICJyaWRlIjogewogICAgICAgICAgICAiUFVMb2NhdGlvbklEIjogMTMwLAogICAgICAgICAgICAiRE9Mb2NhdGlvbklEIjogMjA1LAogICAgICAgICAgICAidHJpcF9kaXN0YW5jZSI6IDMuNjYKICAgICAgICB9LCAKICAgICAgICAicmlkZV9pZCI6IDI1NgogICAgfQ==",
                    "approximateArrivalTimestamp": 1654161514.132
                },
                "eventSource": "aws:kinesis",
                "eventVersion": "1.0",
                "eventID": "shardId-000000000000:49630081666084879290581185630324770398608704880802529282",
                "eventName": "aws:kinesis:record",
                "invokeIdentityArn": "arn:aws:iam::886638369043:role/lambda-kinesis-role",
                "awsRegion": "eu-west-1",
                "eventSourceARN": "arn:aws:kinesis:us-east-1:886638369043:stream/ride_events"
            }
        ]
     }
     ```
     - It will fail if the beginning since the model and other resources are not initialized after only 3 seconds
     - Go to `Configuration -> General configuration -> Edit` and change the following parameter:
       - **Memory**: `256`
       - **Timeout**: `0 min 15 sec`
     - Save the changed parameters
     - Test again. It should work. Otherwise increase the time or memory a little bit, depending on the error.
    
<!-- ## 4.5 Batch - Preparing a scoring script
- Turning the notebook for training the model to a notebook for applying the model
- Turn the notebook into a script
- Clean it and parameterize it -->