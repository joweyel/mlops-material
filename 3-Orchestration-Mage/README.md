# 3. Orchestration and ML Pipelines

- [3.0 Introduction: ML pipelines and Mage](#0-introduction)
- [3.1 Data preparation: ETL and feature engineering](#1-data-preparation)
- [3.2 Training: sklearn models and XGBoost](#2-training)
- [3.3 Observability: Monitoring and alerting](#3-observability)
- [3.4 Triggering: Inference and retraining](#4-triggering)
- [3.5 Deploying: Running operations in production](#5-deploying)
- [3.6 Homework](#6-homework)

<a id="0-introduction"></a>
## 3.0 Introduction: ML pipelines and Mage

### 3.0.1 Machine Learning Pipelines (Workflow Orchestration)
- Transforming a jupyter notebook in something that is easily
    - reproducible
    - runnable
    - parameterized
    - ... 


#### What is a Machine Learning pipeline?
- A sequence of steps that is required to be executed in oder to produce a machine learning model
- Can be achieved by exporting a jupyter notebook, if the code is structured optimally to be exported to a script

![pipeline](imgs/pipeline.jpg)

<u>Exmaple code-skeleton for a Machine Learning pipeline:</u>

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
    model = train_model(X, y, model_parms)
```

Having such a code-skeleton is already better structured than most jupyter notebook code. However it is not yet clear how such a script can be scheduled. Additional to the pipeline-structure, the code needs to be:
- cnetralized somewhere 
- scheduleable
- scalable
- able to retry failed components (often caused by downtimes)

For doing all the steps mentioned abov ususally tools are used instead of dedicated python scipts. Some of those tools are:
- **General purpose workflow orchestration pipeline tools**
    - Airflow (most well known general purpose workflow orchestration tool)
    - Prefect
    - Mage
- **Machine Learning specific workflow orchestration** (usually less flexible but tailored for Machine Learning)
    - Kubeflow
    - MLflow pipelines

#### Running Mage on Linux

Get the repo with the base-code
```bash
git clone https://github.com/mage-ai/mlops.git
```
Run the pipeline
```bash
cd mlops
./scripts/start.sh
```

Open the Mage-UI by opening [http://localhost:6789](http://localhost:6789) in your browser.


<a id="1-data-preparation"></a>
## 3.1 Data preparation: ETL and feature engineering

### 3.1.1 New project

1. **Creating a new Mage project**
![new_project](imgs/3.1_1_new_project.png)

2. **Register the project**
- This is required in a multi-project environment, so that switching between projects is possible
- If the project is not already in the project-list, you have to go to the settings and do it yourself
![setting_project](imgs/3.1_2_setting_project.png)

3. **Create a new pipeline**

- Create the pipeline

![create_pipeline](imgs/3.1_3_create_pipeline.png)

- Rename pipeline and set adequate description

![name_description](imgs/3.1_4_name_description.png)

### 3.1.2 Ingest data

- Creating a Python `Data Loader` block of the name `ingest`, that reads in data from a NYC-Taxi data-file

```python
# ingest.py
import requests
from io import BytesIO
from typing import List

import pandas as pd

if 'data_loader' not in globals():
    from mage_ai.data_preparation.decorators import data_loader

@data_loader
def load_data(*args, **kwargs) -> pd.DataFrame:
    dfs: List[pd.DataFrame] = []

    for year, months in [(2024, (1, 3))]:
        for i in range(*months):
            response = requests.get(
                "https://github.com/mage-ai/datasets/raw/master/taxi/green"
                f"/{year}/{i:02d}.parquet"
            )

            if response.status_code != 200:
                raise Exception(response.text)
            
            df = pd.read_parquet(BytesIO(response.content))
            dfs.append(df)

    return pd.concat(dfs)
```
- Next is to create a time-series chart from the block's outputs

![chart](imgs/3.1_5_chart.png)
![chart_visu](imgs/3.1_6_bar_chart.png)

- To add additional graphs and visualizations you can choose other types when clicking on the graph-symbol right of the play-button of the block
- Having visualizations and data-information at hand can become very handy when using feature-engineering

### 3.1.3 Utility
- Creaing of utility functions, that can be used in different blocks inside a pipeline + creating `__init__.py`-files in `utils` and `data_preparation` folders

![utils_cleaning](imgs/3.1_6_utils_cleaning.png)

```python
import pandas as pd

def clean(
    df: pd.DataFrame,
    include_extreme_durations: bool = False,
) -> pd.DataFrame:
    # Convert pickup and dropoff datetime columns to datetime type
    df.lpep_dropoff_datetime = pd.to_datetime(df.lpep_dropoff_datetime)
    df.lpep_pickup_datetime = pd.to_datetime(df.lpep_pickup_datetime)

    # Calculate the trip duration in minutes
    df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)

    if not include_extreme_durations:
        # Filter out trips that are less than 1 minute or more than 60 minutes
        df = df[(df.duration >= 1) & (df.duration <= 60)]

    # Convert location IDs to string to treat them as categorical features
    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)

    return df
```

All other utility functions can be found here: https://github.com/mage-ai/mlops/tree/master/mlops/utils


### 3.1.4 Prepare
- Creating a `Transformer -> Python -> Base template (generic)` block and name it `prepare`
    - To be able to be used, the function can be parameterized with `**kwargs` as seen in the example below

![prepare_parameterized](imgs/3.1_7_prepare_parameterized.png)

### 3.1.5 Prepare chart

- Getting insights from the data with charts
- Generating charts from the `transformer`-block to look at the distribution of `trip_distance` in form of a histogram. this is done to see if the distribution is skewed or not.

**IMG: TODO**


### 3.1.6 Build Encoders
- An encoder in the Mage-Pipeline ist a function that encodes categorical data with a Scikit-Learn `DictVectorizer`
- The code can also be found in the utils of the Mage MLOps Repo here: https://github.com/mage-ai/mlops/blob/master/mlops/utils/data_preparation/encoders.py

```python
# The code
from typing import Dict, List, Optional, Tuple

import pandas as pd
import scipy
from sklearn.feature_extraction import DictVectorizer

def vectorize_features(
    training_set: pd.DataFrame,
    validation_set: Optional[pd.DataFrame] = None,
) -> Tuple[scipy.sparse.csr_matrix, scipy.sparse.csr_matrix, DictVectorizer]:
    dv = DictVectorizer()

    train_dicts = training_set.to_dict(orient='records')
    X_train = dv.fit_transform(train_dicts)

    X_val = None
    if validation_set is not None:
        val_dicts = validation_set[training_set.columns].to_dict(orient='records')
        X_val = dv.transform(val_dicts)

    return X_train, X_val, dv
```

### 3.1.7 Build code

- Here a data exporter 


<a id="2-training"></a>
## 3.2 Training: sklearn models and XGBoost


<a id="3-observability"></a>
## 3.3 Observability: Monitoring and alerting


<a id="4-triggering"></a>
## 3.4 Triggering: Inference and retraining


<a id="5-deploying"></a>
## 3.5 Deploying: Running operations in production


<a id="6-homework"></a>
## 3.6 Homework
