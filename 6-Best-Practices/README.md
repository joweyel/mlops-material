# 6. Best Practices

- 6.1 [Testing Python code with pytest](#1-pytest)
- 6.2 [Integration tests with docker-compose](#2-integration-test)
- 6.3 [Testing cloud services with LocalStack](#3-local-stack)
- 6.4 [Code quality: linting and formatting](#4-linting)
- 6.5 [Git pre-commit hooks](#5-git)
- 6.6 [Makefiles and make](#6-make)
- 6.7 [Homework](#7-homework)

<a id="1-pytest"></a>
## 6.1 Testing Python code with pytest
In this section the code that was used for predicting on AWS using Kinesis and Lambda from Section 4 of the class, will be tested. The original code can be found here: [streaming](https://github.com/joweyel/mlops-zoomcamp/tree/main/04-deployment/streaming). 

![section4_streaming](imgs/6_1_1_streaming_architecture.jpg)

### Setup

The modified version of the code can be found in the directory [code](./code/).
The goal of this section is to test the code with unit tests.

Everything here is done in the [code](./code/) directory.
 
Installing the *Pipenv* and test-dependencies:
```bash
pipenv install
pipenv install --dev pytest
```
For testing with VS-Code: 
- Install the `Python`-extension
- Select python interpreter with `[Ctrl]+[Shift]+[P]`, then search for `Python: Select Interpreter`
  - *Manually configure the interpreter*
    - Find out path to pipenv: `pipenv --venv`
    - Insert path into `+ Enter interpreter path` and append `/bin/python`
  - *Find Interpreter in Interpreter list (is often already there)*
    - Just click on it!
  - Create `tests` directory in the `code` directory
  
Now open the pipenv and run pytest.
```bash
pipenv shell
pytest
```

### Configure PyTest

The next step is to open the Test-Sidebar in VS-Code (the retort symbol)
- Click on `Configure Python Tests` an select the `tests` directory

Now it's time to test the cpde. The tests can be found in the files of the [`tests`](code/tests/)-directory. While the content from section 4's Streaming folder was initially used it was subsequently adapted to be more suitable to be tested.

The relevant files of this sction are:
- [lambda_function.py](code/lambda_function.py)
- [model.py](code/model.py)
- [model_test.py](code/tests/model_test.py)
- [Dockerfile](code/Dockerfile)
- [test_docker.py](code/test_docker.py)
  

<a id="2-integration-test"></a>
## 6.2 Integration tests with docker-compose

**Types of testing:**
- `Unit-Tests`: Tests a single small "unit" in the code usually functions, methods, etc. by themselves
- `Integration-Tests`: Tests the integration of a code segment into the overall code s.t. the interaction of code with other components works as expected



<a id="3-local-stack"></a>
## 6.3 Testing cloud services with LocalStack

This section handles the testing of cloud ressurces that were left out in the unit- and integration-test sections. For this [localstack](https://github.com/localstack/localstack) is used.

To incorporate `localstack` into the code, `docker-compose` is used. Therefore the previously used docker-compose configuration is extended by the service `kinesis`:

```yaml
services:
  backend:
    image: ${LOCAL_IMAGE_NAME}
    ports:
      - "8080:8080"
    environment:
      - PREDICTIONS_STREAM_NAME=ride_predictions
      - TEST_RUN=True
      - RUN_ID=Test123
      - AWS_DEFAULT_REGION=us-east-1
      - AWS_PROFILE=mlflow-user
      - MODEL_LOCATION=/app/model
    volumes:
      - ~/.aws:/root/.aws
      - ./model:/app/model
  kinesis:
    image: localstack/localstack
    ports: 
      - 4566:4566
    environment:
      - SERVICES=kinesis
```
To pull the `localstack`-image (at first start) and to run the `kinesis`-service the following command has to be used:
```bash
docker-compose up kinesis
```

The goal now is to use local versions of AWS services, that are simulated by `localstack`. To see what `kinesis`-streams are currently running you can use this command:
```bash
aws kinesis list-streams
```
If there are any, you can delete them since they are not used here and are costing you money even in idle-state. If all streams are deleted you should get this result from `aws kinesis list-streams`:
```json
{
    "StreamNames": [],
    "StreamSummaries": []
}
```
Now it's time to connect to the kinesis-stream that was started with the localstack-container.
```bash
aws --endpoint-url=http://localhost:4566 kinesis list-streams
```
If everything worked out, you should get this output:
```json
{
    "StreamNames": []
}
```

Now create a stream with localstack:
```bash
aws --endpoint-url=http://localhost:4566 \
    kinesis create-stream \
    --stream-name ride_predictions \
    --shard-count 1
```

You should now see the strem in the stream list in localstack:
```bash
aws --endpoint-url=http://localhost:4566 kinesis list-streams
{
    "StreamNames": [
        "ride_predictions"
    ]
}
```

<a id="4-linting"></a>
## 6.4 Code quality: linting and formatting

<a id="5-git"></a>
## 6.5 Git pre-commit hooks

<a id="6-make"></a>
## 6.6 Makefiles and make

<a id="7-homework"></a>
## 6.7 Homework
