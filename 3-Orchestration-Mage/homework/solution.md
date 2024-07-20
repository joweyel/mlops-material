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