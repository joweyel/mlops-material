# 3. Orchestration and ML Pipelines

- [3. Orchestration and ML Pipelines](#3-orchestration-and-ml-pipelines)
  - [Machine Learning (ML) Pipelines](#machine-learning-ml-pipelines)
  - [3.1 Introduction to Workflow Orchestration](#31-introduction-to-workflow-orchestration)
    - [If you give an MLOps engineer a job...](#if-you-give-an-mlops-engineer-a-job)
    - [Orchestrate \& observe your Python workflows ar scale](#orchestrate--observe-your-python-workflows-ar-scale)
  - [3.2. Introduction to Prefect](#32-introduction-to-prefect)
    - [Goals of this Section](#goals-of-this-section)
    - [Why use Prefect?](#why-use-prefect)
    - [Self-Hosting a Prefect Server](#self-hosting-a-prefect-server)
    - [Terminology](#terminology)
    - [Example](#example)
  - [3.3 Prefect Workflow](#33-prefect-workflow)
  - [3.4 Deploying your Workflow](#34-deploying-your-workflow)
    - [Create, run and deploy](#create-run-and-deploy)
  - [3.5 Working with Deployments](#35-working-with-deployments)
    - [Creating an Artifact](#creating-an-artifact)
    - [Creating and deploying S3 file with markdown artifact](#creating-and-deploying-s3-file-with-markdown-artifact)
  - [3.6 - Prefect Cloud (Optional)](#36---prefect-cloud-optional)



## Machine Learning (ML) Pipelines

![ml_pipeline](imgs/ml-pipeline.png)

- Usually jupyter notebooks are good for prototyping and visualization of ML results but they are not that suitable in a continuous MLOps setting.
- Notebooks are often transformed to a script and used as a pipeline

**Example (Taxi Data) for a basic ML-Pipeline skeleton:**
```python
def download_data(year, month):
    ...
    return df

def prepare_data(df):
    ...
    return df

def feature_engineering(df):
    ...
    return X, y

def find_best_model(X, y):
    ...
    return params

def train_model(X, y, params):
    ...
    return model

def main():
    df = download_data(2023, 1)
    df = prepare_data(df)
    X, y = feature_engineering(df)
    model_params = find_best_model(X, y)
    model = train_model(X, y, model_params)
```


- **`Prefect-Website`**: [docs.prefect.io](https://docs.prefect.io/v3/get-started)

## 3.1 Introduction to Workflow Orchestration

![worklow](imgs/mlops_workflow.jpg)

- Getting Data from `PostgreSQL`
- Handling the data as `Pandas` dataframe
- Maybe saving the data to a `Parquet` file to use later
- Maybe use `scikit-learn` for feature engineering or running models
- `XGBoost` for running a model
- `MLflow` for experiment tracking
- Using `Flask` for serving the model

There could be failure-points at every connection.

### If you give an MLOps engineer a job...
- Could you set up this pipeline to train this model?
- Could you set up logging?
- Could you do it every day?
- Could you make it retry if it fails?
- Could you send me a messae when it succeeds?
- Could you visualize the dependencies?
- Could you add some caching?
    - With unchanged inputs, we could save some time with this (is however tricky).
- Could you add collaborators to run ad hoc - who don't code?

Those tasks require a lot of work to do and work properly. You could also do everything in `Prefect`.

### Orchestrate & observe your Python workflows ar scale
- `Prefect` providwes tools for working with complex systems, so you can stop wondering about your workflows.
- Learn how to use Prefect to orchestrate and observe your ML workflows

## 3.2. Introduction to Prefect

### Goals of this Section
- Clone Github-Repo
- Setup *Conda*-Environment
- Start `Prefect` server
- Run a Prefect flow
- Checkout Prefect UI

### Why use Prefect?
- Flexible, open-source Python framework to turn standard pipelines into fault tolerant dataflows.
- Installing (on Linux) with `pip install -U prefect`
- For installation on other OS's please look into the [Installation Docs](https://docs.prefect.io/v3/get-started).

### Self-Hosting a Prefect Server
- [Info-Page](https://docs.prefect.io/latest/host/)
- **Orchestration API (REST)**: used by server to work with workflow metadata
- **Database**: stores workflow metadata
- **UI**: visualizes workflows

### Terminology
- **Task**: A discrete unit of work in a Prefect Workflow.
- **Flow**: Container for workflow logic.

```python
from prefect import task, flow

@task
def print_plus_one(obj):
    print(f"Received a {type(obj)} with value {obj}")
    # Shows the type of the parameter after coercion
    print(obj + 1)  # Adds one


# Note that we define the flow with type hints
@flow
def validation_flow(x: int, y: int):
    print_plus_one(x)
    print_plus_one(y)

if __name__ == "__main__":
    validation_flow(1, 2)
```

- **Subflow**: Flow called by another flow

```python
from prefect import task, flow

@task(name="Print Hello")
def print_hello(name):
    msg = f"Hello {name}!"
    print(msg)
    return msg

@flow(name="Subflow")
def my_subflow(msg):
    print(f"Subflow says: {msg}")

@flow(name="Hello Flow")
def hello_world(name="world"):
    message = print_hello(name)
    my_subflow(message)

hello_world("Marvin")
```
The names in the decorators are use in the Prefec UI for visualization.

### Example
1. Get the Repo
```shell
git clone https://github.com/discdiver/prefect-mlops-zoomcamp
cd prefect-mlops-zoomcamp
```
2. Create a conda environment
```shell
conda create -n prefect-ops python=3.10.17
```
3. Install the required dependencies
```shell
pip install -r requirements.txt
# IF there is a problem with pydantic, update `prefect` otherwise don't!
pip install -U prefect
```
4. Start a `Prefect`-Server
```shell
prefect server start
```
5. Configure Prefect to communicate with the server with
```shell
prefect config set PREFECT_API_URL=http://127.0.0.1:4200/api
```
6. Go to the 3.2 sub-folder in the repo, that was previously cloned
7. Open the file `cat_facts.py`, to learn the real facts about cats!
```python
import httpx
from prefect import flow, task

@task(retries=4, retry_delay_seconds=1.0, log_prints=True)
def fetch_cat_fact():
    cat_fact = httpx.get("https://f3-vyx5c2hfpq-ue.a.run.app/")
    #An endpoint that is designed to fail sporadically
    if cat_fact.status_code >= 400:
        raise Exception()
    print(cat_fact.text)

@flow
def fetch():
    fetch_cat_fact()


if __name__ == "__main__":
    fetch()
```
- `@task` decorator of `fetch_cat_fact`:
    - retries the task up to 4 times (in case of failure)
    - time between each try is 1 seconds
    - logs all print statements during the task


8. Look into the UI and see what was saved.
9. Now open `cat_dog_facts.py`. This python file shows how subflows are utilized.
```python
import httpx
from prefect import flow

@flow
def fetch_cat_fact():
    '''A flow that gets a cat fact'''
    return httpx.get("https://catfact.ninja/fact?max_length=140").json()["fact"]

@flow
def fetch_dog_fact():
    '''A flow that gets a dog fact'''
    return httpx.get(
        "https://dogapi.dog/api/v2/facts",
        headers={"accept": "application/json"},
    ).json()["data"][0]["attributes"]["body"]

@flow(log_prints=True)
def animal_facts():
    cat_fact = fetch_cat_fact()
    dog_fact = fetch_dog_fact()
    print(f"🐱: {cat_fact} \n🐶: {dog_fact}")

if __name__ == "__main__":
    animal_facts()
```
- `animal_facts` is the main-flow
    - `fetch_cat_fact` and `fetch_dog_fact` are sub-flows

10. Run `cat_dog_facts.py` (to expand your knowledge about feline mammals) and to see the results in the prefect UI. Lets see them flows!

![prefect_flows](imgs/prefect_flows.png)

## 3.3 Prefect Workflow

**Pipeline without Prefect**
- In the cloned repository open the file [`orchestrate_pre_prefect.py`](prefect-mlops-zoomcamp/3.3/orchestrate_pre_prefect.py) (in sub-folder `3.3`) to see a workflow without prefect.
- You can run the pipeline from the root-directory of the repo with `python3 3.3/orchestrate_pre_prefect.py`

The data-files used are also in the repository, but they could be from another year, that is not in the default-parameters of the main-function. You have to add the new paths yourself.

- Now open [`orchestrate.py`](prefect-mlops-zoomcamp/3.3/orchestrate.py) from the same folder `3.3`. This file has the adequate `prefect`-decorators
```python
# retries opening 3 times + delay for when data is obtained from the internet
@task(retries=3, retry_delay_seconds=2)
def read_data(filename: str) -> pd.DataFrame:
    ...

# simple task with no additional parameters
@task
def add_features(
        df_train: pd.DataFrame, df_val: pd.DataFrame
):
    ...

# logs all console outputs
@task(print_logs=True)
def train_best_model(...):
    ...
```
- Run the programm and go to the UI and see the results.

## 3.4 Deploying your Workflow
- Productionizing the workflow
- Deploy on Prefect Server (locally for now)
- Enables scheduling and team collaboration (especially on Prefect Cloud)

### Create, run and deploy

<!-- <p align="center">
    <img src="imgs/Activity-create-run-deployment.png" style="max-width: 80%; height: auto;">
    <figcaption align="center">Think it, dream it, do it!</figcaption>
</p> -->

1. **Ensure your directory is set up**
    - Navigate to the `prefect-mlops-zoomcamp` directory
    - Add `@flow` and `@task` decorators to your Python scripts as needed
    - There is no longer a `prefect project init` command in Prefect 3

2. **Start the Prefect Server (locally)**
    ```shell
    prefect server start
    ```

3. **Create a Work Pool**
    - Can be done via the Prefect UI or CLI
    - Example using CLI:
    ```shell
    prefect work-pool create zoompool -t process
    ```

4. **Deploy the flow**
    - Use `prefect deploy` to create a deployment YAML configuration
    - Specify the entrypoint to your flow as `file.py:flow_function`
    - Example:
    ```shell
    prefect deploy 3.4/orchestrate.py:main_flow -n taxi_local_data -p zoompool
    ```
    - You will be prompted:
        - "Would you like your workers to pull your flow code from a remote storage location?" → answer `n`
        - "Would you like to configure schedules?" → answer `n`
        - "Would you like to save configuration for this deployment?" → answer `y`

5. **Start a worker to poll the Work Pool**
    ```shell
    prefect worker start -p zoompool
    ```

6. **Run the deployment**
    - From CLI, trigger a run manually:
    ```shell
    prefect deployment run main-flow/taxi_local_data --param train_path=./data/green_tripdata_2023-01.parquet --param val_path=./data/green_tripdata_2023-02.parquet
    ```

7. **What about `prefect.yaml`?**
    - This file is generated automatically by `prefect deploy`
    - You can customize it as needed
    - Example structure:
    ```yaml
    name: taxi_local_data
    version: null
    tags: []
    description: null
    schedule: null
    parameters:
      train_path: "./data/green_tripdata_2023-01.parquet"
      val_path: "./data/green_tripdata_2023-02.parquet"
    flow_name: main_flow
    entrypoint: orchestrate.py:main_flow
    work_pool:
      name: zoompool
      work_queue_name: null
      job_variables: {}
    ```

![deploy](imgs/prefect_run.png)



## 3.5 Working with Deployments
- In this section the connection with AWS S3 Buckets is shown
- You need to install the `aws`-version of `prefect` to use the required functionality
```shell
# Simple as!
pip install prefect-aws
```
- To look into the documentation for `prefect-aws` please look into the Documentation [here](https://prefecthq.github.io/prefect-aws/).
- Go to your AWS account and create a new `S3 Bucket` for this task. The bucket is access-blocked by default.
    - If you dont already have a `IAM-User` with S3 access, please create one
    - When the IAM user is created, you can create Access keys
    - The generated access keys can now be used in the code of `create_s3_bucket_block.py`.

```python
import os
from time import sleep
from prefect_aws import S3Bucket, AwsCredentials

## IMPORTANT: Dont hardcode the credentials in the program
##            Rather use environment variables.
AWS_ACCESS_KEY_ID = os.environ['AWS_ACCESS_KEY_ID']
AWS_SECRET_ACCESS_KEY = os.environ['AWS_SECRET_ACCESS_KEY']

def create_aws_creds_block():
    my_aws_creds_obj = AwsCredentials(
        aws_access_key_id=AWS_ACCESS_KEY_ID, 
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY
    )
    my_aws_creds_obj.save(name="my-aws-creds", overwrite=True)

def create_s3_bucket_block():
    aws_creds = AwsCredentials.load("my-aws-creds")
    my_s3_bucket_obj = S3Bucket(
        bucket_name="zoomcamp-mlops-prefect-bucket", credentials=aws_creds
    )
    my_s3_bucket_obj.save(name="s3-bucket-example", overwrite=True)


if __name__ == "__main__":
    create_aws_creds_block()
    sleep(5)
    create_s3_bucket_block()
```

When everything is setup you can call the following code to create the credentials and the bucket block:
```shell
python3 3.5/create_s3_bucket_block.py
```

The resulting blocks can be seen here:

![prefect_blocks](imgs/prefect_blocks.png)

To see all current blocks with the command line you have to call the following command:
```shell
# List all blocks
prefect blocks ls
# List Block Types that are available
prefect block type ls
```
To be able to use created blocks and send them to the server, you have to register them. This tells the server that all the blocks are available.
```shell
prefect block register -m prefect_aws

# Console Output:
Successfully registered 5 blocks

┏━━━━━━━━━━━━━━━━━━━┓
┃ Registered Blocks ┃
┡━━━━━━━━━━━━━━━━━━━┩
│ AWS Credentials   │
│ AWS Secret        │
│ ECS Task          │
│ MinIO Credentials │
│ S3 Bucket         │
└───────────────────┘

 To configure the newly registered blocks, go to the 
Blocks page in the Prefect UI: 
http://127.0.0.1:4200/blocks/catalog
```
The server now knows that those 5 types of blocks exist. If you go to the prefect server and want to add a new block, go to the `blocks`-tab and add the blocks you like.

### Creating an Artifact

The code can be fonud here: [orchestrate_s3.py](https://github.com/discdiver/prefect-mlops-zoomcamp/blob/main/3.5/orchestrate_s3.py)
Changes in comparison to `orchestrate.py`, which works locally:
- Accesses the data from an S3 bucket
  ```python
  # Additional imports
  from prefect_aws import S3Bucket
  from prefect.artifacts import create_markdown_artifact
  ...

  # In train_best_model(...)
  markdown__rmse_report = f"""# RMSE Report

  ## Summary

  Duration Prediction 

  ## RMSE XGBoost Model

  | Region    | RMSE |
  |:----------|-------:|
  | {date.today()} | {rmse:.2f} |
  """

  create_markdown_artifact(
      key="duration-model-report", markdown=markdown__rmse_report
  )

  ... 
  # Load in main_flow_s3(...)
  s3_bucket_block = S3Bucket.load("s3-bucket-block")
  s3_bucket_block.download_folder_to_path(from_folder="data", to_folder="data")
  ```
  - This will download the requested data from the S3 bucket and place it in the `data` folder.
  - Now the model will be trained and the artifacts will be generated
  ![artifacts](imgs/artifacts.png)

### Creating and deploying S3 file with markdown artifact
- Go to the deployment.yaml file in the main-directory of the cloned github-repo
```yaml
# deployment.yaml
deployments:
- name: taxi_local_data
  entrypoint: 3.4/orchestrate.py:main_flow
  work_pool: 
    name: zoompool
- name: taxi_s3_data
  entrypoint: 3.5/orchestrate_s3.py:main_flow_s3
  work_pool: 
    name: zoompool
```
- This file contains 2 different deployments, that rould be run
    - To run all deployments use
    ```shell
    prefect deploy --all
    ```
    - To run a single deployment use
    ```shell
    prefect deploy -n <deployment-name>
    ```
- The rendered Markdown

```markdown
## Summary

Duration Prediction 

## RMSE XGBoost Model

| Region    | RMSE |
|:----------|-------:|
| 2023-06-16 | 5.20 |
```

## 3.6 - Prefect Cloud (Optional)

TODO