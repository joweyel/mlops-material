## Question 1. Run Mage
What's the version of Mage we run? 

- **Solution:** `v0.9.72`


## Question 2. Creating a project

Now let's create a new project. We can call it "homework_03", for example.

How many lines are in the created `metadata.yaml` file? 

- **Solution:** `55`


## Question 3. Creating a pipeline

Let's create an ingestion code block.

In this block, we will read the March 2023 Yellow taxi trips data.

How many records did we load? 

- **Solution:** `3,403,766`


## Question 4. Data preparation

Let's use the same logic for preparing the data we used previously. We will need to create a transformer code block and put this code there.

This is what we used (adjusted for yellow dataset):

```python
def read_dataframe(filename):
    df = pd.read_parquet(filename)

    df.tpep_dropoff_datetime = pd.to_datetime(df.tpep_dropoff_datetime)
    df.tpep_pickup_datetime = pd.to_datetime(df.tpep_pickup_datetime)

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df.duration = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    
    return df
```

Let's adjust it and apply to the data we loaded in question 3. 

What's the size of the result? 

- **Solution:** `3,316,216`

<details>
<summary><b>Code:</b> data_preparation.py</summary>

```python
import pandas as pd

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer

@transformer
def transform(data, *args, **kwargs):

    data["tpep_dropoff_datetime"] = pd.to_datetime(data["tpep_dropoff_datetime"])
    data["tpep_pickup_datetime"] = pd.to_datetime(data["tpep_pickup_datetime"])

    data["duration"] = data["tpep_dropoff_datetime"] - data["tpep_pickup_datetime"]
    data["duration"] = data["duration"].dt.total_seconds() / 60

    data = data[(data["duration"] >= 1) & (data["duration"] <= 60)]

    categorical = ["PULocationID", "DOLocationID"]
    data[categorical] = data[categorical].astype(str)

    return data
```
</details>


## Question 5. Train a model

We will now train a linear regression model using the same code as in homework 1.

* Fit a dict vectorizer.
* Train a linear regression with default parameters.
* Use pick up and drop off locations separately, don't create a combination feature.

Let's now use it in the pipeline. We will need to create another transformation block, and return both the dict vectorizer and the model.

What's the intercept of the model? 

- **Solution:** `24.77`
<details>
<summary><b>train.py</b></summary>

```python
import pandas as pd

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@transformer
def train(df, *args, **kwargs):

    categorical = ["PULocationID", "DOLocationID"]
    target = "duration"

    train_dicts = df[categorical].to_dict(orient="records")

    dv = DictVectorizer().fit(train_dicts)
    X_train = dv.transform(train_dicts)
    y_train = df[target].values

    model = LinearRegression()
    model.fit(X_train, y_train)

    print(model.intercept_)

    return dv, model


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'
```

</details>


## Question 6. Register the model 

The model is trained, so let's save it with MLFlow.

If you run mage with docker-compose, stop it with Ctrl+C or 

```bash
docker-compose down
```

Let's create a dockerfile for mlflow, e.g. `mlflow.dockerfile`:

```dockerfile
FROM python:3.10-slim

RUN pip install mlflow==2.12.1

EXPOSE 5000

CMD [ \
    "mlflow", "server", \
    "--backend-store-uri", "sqlite:///home/mlflow_data/mlflow.db", \
    "--host", "0.0.0.0", \
    "--port", "5000" \
]
```

And add it to the docker-compose.yaml:

```yaml
  mlflow:
    build:
      context: .
      dockerfile: mlflow.dockerfile
    ports:
      - "5000:5000"
    volumes:
      - "${PWD}/mlflow_data:/home/mlflow_data/"
    networks:
      - app-network
```

Note that `app-network` is the same network as for mage and postgres containers.
If you use a different compose file, adjust it.

We should already have `mlflow==2.12.1` in requirements.txt in the mage project we created for the module. If you're starting from scratch, add it to your requirements.

Next, start the compose again and create a data exporter block.

In the block, we

* Log the model (linear regression)
* Save and log the artifact (dict vectorizer)

If you used the suggested docker-compose snippet, mlflow should be accessible at `http://mlflow:5000`.

Find the logged model, and find MLModel file. What's the size of the model? (`model_size_bytes` field)

- **Solution:** `4,534`

<details>
<summary><b>save.py</b></summary>

```python
import pickle
import mlflow

mlflow.set_tracking_uri("http://mlflow:5000")
mlflow.set_experiment("nyc-taxi-experiment")


if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter


@data_exporter
def export_data(data, *args, **kwargs):
    dv, model = data

    with mlflow.start_run():
        with open("dict_vectorizer.bin", "wb") as f_out:
            pickle.dump(dv, f_out)
        mlflow.log_artifact("dict_vectorizer.bin")
        mlflow.sklearn.log_model(model, "model")

    print("Absolutely LOGGED!")
```
</details>
