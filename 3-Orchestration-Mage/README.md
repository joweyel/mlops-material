# 3. Orchestration and ML Pipelines

- [3. Orchestration and ML Pipelines](#3-orchestration-and-ml-pipelines)
  - [3.0 Introduction: ML pipelines and Mage](#30-introduction-ml-pipelines-and-mage)
    - [3.0.1 Machine Learning Pipelines (Workflow Orchestration)](#301-machine-learning-pipelines-workflow-orchestration)
      - [What is a Machine Learning pipeline?](#what-is-a-machine-learning-pipeline)
      - [Running Mage on Linux](#running-mage-on-linux)
  - [3.1 Data preparation: ETL and feature engineering](#31-data-preparation-etl-and-feature-engineering)
    - [3.1.1 New project](#311-new-project)
    - [3.1.2 Ingest data](#312-ingest-data)
    - [3.1.3 Utility](#313-utility)
    - [3.1.4 Prepare](#314-prepare)
    - [3.1.5 Prepare chart](#315-prepare-chart)
    - [3.1.6 Build Encoders](#316-build-encoders)
    - [3.1.7 Build code \& 3.1.8 Build test](#317-build-code--318-build-test)
  - [3.2 Training: sklearn models and XGBoost](#32-training-sklearn-models-and-xgboost)
    - [3.2.1 GDP training set](#321-gdp-training-set)
    - [3.2.2 Sklearn training GDP](#322-sklearn-training-gdp)
    - [3.2.3 Load models](#323-load-models)
    - [3.2.4 Load models utility](#324-load-models-utility)
    - [3.2.5 Hyperparameter tuning](#325-hyperparameter-tuning)
    - [3.2.6 Sklearn trained](#326-sklearn-trained)
    - [3.2.7 Hyperparameter XGBoost](#327-hyperparameter-xgboost)
    - [3.2.8 XGBoost trained](#328-xgboost-trained)
  - [3.3 Observability: Monitoring and alerting](#33-observability-monitoring-and-alerting)
    - [3.3.1 Sklearn](#331-sklearn)
    - [3.3.2 XGBoost part 1 \& 3.3.3 XGBoost part 2](#332-xgboost-part-1--333-xgboost-part-2)
    - [3.3.4 Overview](#334-overview)
    - [3.3.5 Time series bar](#335-time-series-bar)
    - [3.3.6 Histogram](#336-histogram)
    - [3.3.7 Bar chart](#337-bar-chart)
    - [3.3.8 Pie chart](#338-pie-chart)
    - [3.3.9 Setup alerts](#339-setup-alerts)
    - [3.3.10 Email](#3310-email)
  - [3.4 Triggering: Inference and retraining](#34-triggering-inference-and-retraining)
    - [3.4.1 Retraining pipeline](#341-retraining-pipeline)
    - [3.4.2 Trigger](#342-trigger)
    - [3.4.3 Predict](#343-predict)
    - [3.4.5 Inference notebook](#345-inference-notebook)
    - [3.4.6 Interactions \& 3.4.7 Interactions run](#346-interactions--347-interactions-run)
    - [3.4.8 API](#348-api)
  - [3.5 Deploying: Running operations in production](#35-deploying-running-operations-in-production)
    - [3.5.1 Setup email](#351-setup-email)
    - [3.5.2 Deploying： Deploy](#352-deploying-deploy)
    - [3.5.3 CI⧸CD](#353-cicd)
    - [3.5.4 Deployed](#354-deployed)
  - [3.6 Homework](#36-homework)


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

<details>
<summary><b>ingest.py</b></summary>

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
</details>

- Next is to create a time-series chart from the block's outputs

![chart](imgs/3.1_5_chart.png)
![chart_visu](imgs/3.1_6_bar_chart.png)

- To add additional graphs and visualizations you can choose other types when clicking on the graph-symbol right of the play-button of the block
- Having visualizations and data-information at hand can become very handy when using feature-engineering

### 3.1.3 Utility
- Creaing of utility functions, that can be used in different blocks inside a pipeline + creating `__init__.py`-files in `utils` and `data_preparation` folders

![utils_cleaning](imgs/3.1_6_utils_cleaning.png)

<details>
<summary><b>prepare.py</b></summary>

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
</details>

All other utility functions can be found here: https://github.com/mage-ai/mlops/tree/master/mlops/utils


### 3.1.4 Prepare
- Creating a `Transformer -> Python -> Base template (generic)` block and name it `prepare`
    - To be able to be used, the function can be parameterized with `**kwargs` as seen in the example below

![prepare_parameterized](imgs/3.1_7_prepare_parameterized.png)

### 3.1.5 Prepare chart

- Getting insights from the data with charts
- Generating charts from the `transformer`-block to look at the distribution of `trip_distance` in form of a histogram. this is done to see if the distribution is skewed or not.

![3_1_5_histogram](imgs/3_1_5_histogram.png)

```python
# custom code for plotting
import pandas as pd

from mage_ai.shared.parsers import convert_matrix_to_dataframe


if isinstance(df_1, list) and len(df_1) >= 1:
    item = df_1[0]
    if isinstance(item, pd.Series):
        item = item.to_frame()
    elif not isinstance(item, pd.DataFrame):
        item = convert_matrix_to_dataframe(item)
    df_1 = item

columns = df_1.columns
col = "trip_distance"
x = df_1[df_1[col] <= 20][col]
```


### 3.1.6 Build Encoders
- An encoder in the Mage-Pipeline is a function that encodes categorical data with a Scikit-Learn `DictVectorizer`, that was used 
- The code can also be found in the utils of the Mage MLOps Repo here: https://github.com/mage-ai/mlops/blob/master/mlops/utils/data_preparation/encoders.py

<details>
<summary><b>encoder.py</b></summary>

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
</details>

### 3.1.7 Build code & 3.1.8 Build test

- Here a data exporter vectorizes the code, selects relevant features and returns the data

<details>
<summary><b>builder.py</b></summary>

```python
from typing import Tuple
from pandas import DataFrame, Series
from scipy.sparse._csr import csr_matrix
from sklearn.base import BaseEstimator

from mlops.utils.data_preparation.encoders import vectorize_features
from mlops.utils.data_preparation.feature_selector import select_features

if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter

if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test

@data_exporter
def export(
    data: Tuple[DataFrame, DataFrame, DataFrame], 
    **kwargs
) -> Tuple[csr_matrix, csr_matrix, csr_matrix, Series, Series, Series, BaseEstimator
]:
    df, df_train, df_val = data
    target = kwargs.get('target', 'duration')
    X, _, _ = vectorize_features(select_features(df))
    y: Series = df[target]

    X_train, X_val, dv = vectorize_features(
        select_features(df_train), 
        select_features(df_val)
    )
    y_train = df_train[target]
    y_val = df_val[target]

    return X, X_train, X_val, y, y_train, y_val, dv

# Tests for correct number of rows and features
@test
def test_dataset(
    X: csr_matrix,
    X_train: csr_matrix,
    X_val: csr_matrix,
    y: Series,
    y_train: Series,
    y_val: Series,
    *args
) -> None:
    assert (
        X.shape[0] == 105870
    ), f"Entire dataset should have 105870 examples, but has {X.shape[0]}"
    assert (
        X.shape[1] == 7027
    ), f"Entire dataset should have 7027 features, but has {X.shape[1]}"
    assert (
        len(y.index) == X.shape[0]
    ), f"Entire dataset should have {X.shape[0]} examples, but has {len(y.index)}"

```
</details>

<a id="2-training"></a>
## 3.2 Training: sklearn models and XGBoost

### 3.2.1 GDP training set

- A Global Data Product (GDP) is a reusable data entity that can be utilized across multiple pipelines. 
  
<p float="left">
  <img src="imgs/3_2_1_gdp1.png" width="45%" />
  <img src="imgs/3_2_1_gdp2.png" width="45%" />
</p>

### 3.2.2 Sklearn training GDP
- Creating training-pipeline for sklearn models of name `sklearn_training`
- Click `Add blocks` and choose `Global data product`, then select `training_set`

### 3.2.3 Load models
- Create custom Python block that loads all Sklearn models of name `load_models`

<details>
<summary><b>load_models.py</b></summary>

```python
from typing import Dict, List, Tuple

if 'custom' not in globals():
    from mage_ai.data_preparation.decorators import custom

@custom
def models(*args, **kwargs) -> Tuple[List[str], List[Dict[str, str]]]:
    """
    models: comma seperated strings
        linear_model.Lasso
        linear_model.LinearRegression
        svm.LinearSVM
        ensemble.ExtraTreesRegressor
        ensemble.GradientBoostingRegressor
        ensemble.RandomForestRegressor
    """
    model_names: str = kwargs.get(
        "models", "linear_model.LinearRegression,linear_model.Lasso"
    )
    child_data: List[str] = [
        model_name.strip() for model_name in model_names.split(",")
    ]
    child_metadata: List[Dict] = [
        dict(block_uuid=model_name.split(".")[-1]) for model_name in child_data
    ]
    return child_data, child_metadata
```
</details>

- Setting the `load_model` block as dynamic (option in the circle with the three dots)

### 3.2.4 Load models utility
- Remove connection from `GDP` to the `load_models`-block
- Run the `load_models`-block. You will see the 2 default models defined in the code
- Create and use utility functions for model training and hyperparameter tuning
  - Hyperparameter tuning utils [here](mlops/mlops/utils/hyperparameters/)
  - Model utils [here](mlops/mlops/utils/models/)

### 3.2.5 Hyperparameter tuning
- Create `Transformer`-block of type `Base template (generic)`, Python as language and name it `hyperparameter_tuning/sklearn`
- Connect the `training_set` GDP to it
- The GDP `training_set` has to be the first input and the `load_model`-block has to be the second input to have the correct order of parameters in the `hyperparameter_tuning/sklearn`-block

<details>
<summary><b>hyperparameter_tuning.py</b></summary>

```python
from typing import Callable, Dict, Tuple, Union

from pandas import Series
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator

from mlops.utils.models.sklearn import load_class, tune_hyperparameters

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer

@transformer
def hyperparameter_tuning(
    training_set: Dict[str, Union[Series, csr_matrix]],
    model_class_name: str,
    *args,
    **kwargs
) -> Tuple[
    Dict[str, Union[bool, float, int, str]],
    csr_matrix,
    Series,
    Callable[..., BaseEstimator]
]:
    X, X_train, X_val, y, y_train, y_val, _ = training_set['build']

    model_class = load_class(model_class_name)
    
    hyperparameters = tune_hyperparameters(
        model_class,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        max_evaluations=kwargs.get('max_evaluations', 1),
        random_state=kwargs.get('random_state', 3)
    )
    return hyperparameters, X, y, dict(cls=model_class, name=model_class_name)
```
</details>

### 3.2.6 Sklearn trained
- After finding the best hyperparameters for a trained model, the next step is to train a final model with these hyperparameters
- Zhe final model is returned on the whole dataset
- Create `Data exporter`-block of type `Base template (generic)`

<details>
<summary><b>sklearn.py</b></summary>

```python
from typing import Callable, Dict, Tuple, Union

from pandas import Series
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator

from mlops.utils.models.sklearn import load_class, train_model

if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter


@data_exporter
def train(
    settings: Tuple[
        Dict[str, Union[bool, float, int, str]],
        csr_matrix,
        Series,
        Dict[str, Union[Callable[..., BaseEstimator]]],
    ],
    **kwargs
) -> Tuple[BaseEstimator, Dict[str, str]]:
    hyperparameters, X, y, model_info = settings

    model_class = model_info["cls"]
    model = model_class(**hyperparameters)
    model.fit(X, y)

    return model, model_info
```
</details>

### 3.2.7 Hyperparameter XGBoost
- Creating new pipeline (`xgboost_training`) for hyperparameter-tuning and training of XGBoost model
- Add GDP `training_set` to the pipeline
- Create `transformer`-block of type `Base template (generic)` and name it `hyperparameter_tuning/XGBoost`
- <u>Utility functions for XGBoost hyperparameter-tuning</u> can be found here https://github.com/mage-ai/mlops/tree/master/mlops/utils/hyperparameters/shared.py
- <u>Utility functions for XGBoost models</u> can be found in `xgboost.py` here https://github.com/mage-ai/mlops/tree/master/mlops/utils/models/xgboost.py
- <u>Utility functions for Logging of Experiments</u> can be found in `logging.py` here https://github.com/mage-ai/mlops/blob/master/mlops/utils/logging.py
  - For Logging MLflow is used
- **Variables** can be used to configure parameters like `early_stopping_rounds` and `max_evaluations`

<details>
<summary><b>hperparameter_tuning/xgboost.py</b></summary>

```python
from typing import Dict, Tuple, Union

import numpy as np
import xgboost as xgb
from pandas import Series
from scipy.sparse import csr_matrix

from mlops.utils.logging import track_experiment
from mlops.utils.models.xgboost import build_data, tune_hyperparameters

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@transformer
def hyperparameter_tuning(
    training_set: Dict[str, Union[Series, csr_matrix]],
    **kwargs
) -> Tuple[
    Dict[str, Union[bool, float, int, str]],
    csr_matrix,
    Series
]:
    X, X_train, X_val, y, y_train, y_val, _ = training_set["build"]
    training = build_data(X_train, y_train)
    validation = build_data(X_val, y_val)

    best_hyperparameters = tune_hyperparameters(
        training, 
        validation,
        callback=lambda **opts: track_experiment(**{**opts, **kwargs}),
        **kwargs
    )

    return best_hyperparameters, X_train, y_train
```
</details>

### 3.2.8 XGBoost trained

- Obtaining best hyperparameters with global variables `early_stopping_rounds=3` and `max_evaluations=3`
- Creating a `Data exporter`-block with the name of `XGBoost` (`xgboost.py`)
- Using variable `max_depth=10` for testing the code but should be increased for production

<details>
<summary><b>xgboost.py</b></summary>

```python
from typing import Dict, Tuple, Union

from pandas import Series
from scipy.sparse._csr i mport csr_matrix
from xgboost import Booster

from mlops.utils.models.xgboost import build_data, fit_model

if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter


@data_exporter
def export_data(
    training_set: Dict[str, Union[Series, csr_matrix]],
    settings: Tuple[
        Dict[str, Union[bool, float, int, str]],
        csr_matrix,
        Series,
    ],
    **kwargs
) -> Tuple[Booster, csr_matrix, Series]:
    hyperparameters, X, y = settings

    # Test training a model with low max_depth
    # so that the output renderes a reasonably sized plot tree.
    if kwargs.get("max_depth"):
        hyperparameters["max_depth"] = int(kwargs.get("max_depth"))

    model = fit_model(
        build_data(X, y),
        hyperparameters,
        verbose_eval=kwargs.get("verbose_eval", 100),
    )

    # DictVectorizer to transform features for online inference.
    vectorizer = training_set["build"][6]
    return model, vectorizer

```
</details>


<a id="3-observability"></a>
## 3.3 Observability: Monitoring and alerting

### 3.3.1 Sklearn
- Open `unit_3_observability` and open `skelearn_training` pipeline
- Go to `Dashboard` and click on `Add recommended charts` if there are no charts yet
- Create a trigger of type `API` and create it
  
### 3.3.2 XGBoost part 1 & 3.3.3 XGBoost part 2
- Open `xgboost_training` pipeline
- Add a `Custom`-block of Python type with the name `dashboard_data_source`
- The outputs of `xgboost`-block should be the first input `training_results` and the output of `hyperparameter_tuning/xgboost`-block should be the second input `settings`
- The data obtained from here will be displayed in charts

<details>
<summary><b>dashboard_data_source.py</b></summary>

```python
from typing import Dict, Tuple, Union

from pandas import Series
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator
from xgboost import Booster

if 'custom' not in globals():
    from mage_ai.data_preparation.decorators import custom


@custom
def source(
    training_results: Tuple[Booster, BaseEstimator],
    settings: Tuple[
        Dict[str, Union[bool, float, int, str]],
        csr_matrix,
        Series,
    ],
    **kwargs,
) -> Tuple[Booster, csr_matrix, csr_matrix]:
    model, _ = training_results
    _, X_train, y_train = settings

    return model, X_train, y_train
```
</details>

**Code for all custom charts can be found here**: [unit_3_observability/charts](mlops/mlops/unit_3_observability/charts/)

### 3.3.4 Overview
- How to create different types of visualization.
- Please watch the videos 3.3.4 to 3.3.10 from this [playlist](https://www.youtube.com/playlist?list=PL3MmuxUbc_hIUISrluw_A7wDSmfOhErJK). Each is about 2 to 3 minutes long.

### 3.3.5 Time series bar
[Video](https://www.youtube.com/watch?v=6kqHoxAL0DY&list=PL3MmuxUbc_hIUISrluw_A7wDSmfOhErJK&index=38&pp=iAQB)


### 3.3.6 Histogram
[Video](https://www.youtube.com/watch?v=GQMgCzI-Qrg&list=PL3MmuxUbc_hIUISrluw_A7wDSmfOhErJK&index=39&pp=iAQB)


### 3.3.7 Bar chart
[Video](https://www.youtube.com/watch?v=q4Quk6GeVRk&list=PL3MmuxUbc_hIUISrluw_A7wDSmfOhErJK&index=40&pp=iAQB)


### 3.3.8 Pie chart
[Video](https://www.youtube.com/watch?v=I5qR3OtASXs&list=PL3MmuxUbc_hIUISrluw_A7wDSmfOhErJK&index=41&pp=iAQB)


### 3.3.9 Setup alerts
[Video](https://www.youtube.com/watch?v=DjtE3webtjE&list=PL3MmuxUbc_hIUISrluw_A7wDSmfOhErJK&index=42&pp=iAQB)


### 3.3.10 Email
[Video](https://www.youtube.com/watch?v=H6D7zyqSQMw&list=PL3MmuxUbc_hIUISrluw_A7wDSmfOhErJK&index=43&pp=iAQB)


<a id="4-triggering"></a>
## 3.4 Triggering: Inference and retraining

### 3.4.1 Retraining pipeline
- Create new batch pipeline with the name `automatic_retraining`
  - Set tag `retraining`
- Add `Sensor`-block of type `Base template (generic)` and name it `detect_new_data`
  - Checks if new data has arrived in the dataset and re-trains the models with the new (bigger) dataset

<details>
<summary><b>detect_new_data.py</b></summary>

```python
import json
import os
import requests

from mage_ai.settings.repo import get_repo_path

if 'sensor' not in globals():
    from mage_ai.data_preparation.decorators import sensor


@sensor
def check_for_new_data(*args, **kwargs) -> bool:
    path = os.path.join(get_repo_path(), '.cache', 'data_tracker')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    data_tracker_prev = {}
    if os.path.exists(path):
        with open(path, 'r') as f:
            data_tracker_prev = json.load(f)

    data_tracker = requests.get('https://hub.docker.com/v2/repositories/mageai/mageai').json()
    with open(path, 'w') as f:
        f.write(json.dumps(data_tracker))

    count_prev = data_tracker_prev.get('pull_count')
    count = data_tracker.get('pull_count')
    
    print(f'Previous count: {count_prev}')
    print(f'Current count:  {count}')

    should_train = count_prev is None or count > count_prev
    if should_train:
        print('Retraining models...')
    else:
        print('Not enough new data to retrain models.')
    
    return should_train
```
</details>

**Re-Training Sklearn Model**

- Add `Custom`-block of type `Python block` and name it `retrain/sklearn`
  - This block should trigger the training pipeline `sklearn_training` of the sklearn model
  - The output of the `Sensor`-block `detect_new_data` is the input of `retrain/sklearn`
  
<details>
<summary><b>retrain/sklearn.py</b></summary>

```python
from mage_ai.orchestration.triggers.api import trigger_pipeline

if 'custom' not in globals():
    from mage_ai.data_preparation.decorators import custom


@custom
def retrain(*args, **kwargs):
    models = [
        'linear_model.Lasso',
        'linear_model.LinearRegression',
        'svm.LinearSVR',
        'ensemble.ExtraTreesRegressor',
        'ensemble.GradientBoostingRegressor',
        'ensemble.RandomForestRegressor',
    ]

    trigger_pipeline(
        'sklearn_training',
        check_status=True,
        error_on_failure=True,
        schedule_name='Automatic retraining for sklearn models',
        verbose=True,
    )
```
</details>

**Re-Training of XGBoost Model**

- Add `Custom`-block of type `Python block` and name it `retrain/xgboost`
  - This block should trigger the training pipeline `xgboost_training` of the xgboost model
  - The output of the `Sensor`-block `detect_new_data` is the input of `retrain/xgboost`

<details>
<summary><b>retrain/xgboost.py</b></summary>

```python
from mage_ai.orchestration.triggers.api import trigger_pipeline

if 'custom' not in globals():
    from mage_ai.data_preparation.decorators import custom


@custom
def retrain(*args, **kwargs):
    trigger_pipeline(
        'xgboost_training',
        check_status=True,
        error_on_failure=True,
        schedule_name='Automatic retraining for XGBoost',
        verbose=True,
    )
```
</details>

### 3.4.2 Trigger
- Create a trigger that runs the `automatic_retraining` pipeline on a regular schedule

![trigger](imgs/3_4_2_Trigger_1.png)

- The created trigger will be saved besides the pipeline in the `pipelines`-directory. The trigger will be created on another computer when the pipeline data is read in (if not already there).

```yaml
# In pipelines/automatic_retraining/
triggers:
- description: null
  envs: []
  last_enabled_at: 2024-07-19 16:50:37.549256+00:00
  name: Retrain models hourly
  pipeline_uuid: automatic_retraining
  schedule_interval: '@hourly'
  schedule_type: time
  settings:
    allow_blocks_to_fail: true
    create_initial_pipeline_run: true
    skip_if_previous_running: true
  sla: null
  start_time: 2024-05-16 07:00:00+00:00
  status: active
  token: c8361973c2b742199f20423ac2c2d123
  variables: {}
```

### 3.4.3 Predict
- Create batch pipeline for online inference with the name `predict`
- Create a `Markdown`-block with instructions on how to send data to the pipeline

```md
# Sample online inference

Use the following CURL command to get real-time predictions:

curl --location 'http://localhost:6789/api/runs' \
--header 'Authorization: Bearer ef46ef6d89f44853b3d31ef771991bc5' \
--header 'Content-Type: application/json' \
--header 'Cookie: lng=en' \
--data '{
    "run": {
        "pipeline_uuid": "predict",
        "block_uuid": "inference",
        "variables": {
            "inputs": [
                {
                    "DOLocationID": "239",
                    "PULocationID": "236",
                    "trip_distance": 1.98
                },
                {
                    "DOLocationID": "170",
                    "PULocationID": "65",
                    "trip_distance": 6.54
                }
            ]
        }
    }
}'

## Note

The `Authorization` header is using this pipeline’s API trigger’s token value.
The token value is set to `fire` for this project.
If you create a new trigger, that token will change.
Only use a fixed token for testing or demonstration purposes.
```

- Create `Custom`-block and name it `inference`

<details>
<summary><b>inference.py</b></summary>

```python
from typing import Dict, List, Tuple, Union

from sklearn.feature_extraction import DictVectorizer
from xgboost import Booster

from mlops.utils.data_preparation.feature_engineering import combine_features
from mlops.utils.models.xgboost import build_data

if 'custom' not in globals():
    from mage_ai.data_preparation.decorators import custom

DEFAULT_INPUTS = [
    {
        # target = "duration": 11.5
        'DOLocationID': 239,
        'PULocationID': 236,
        'trip_distance': 1.98,
    },
    {
        # target = "duration" 20.8666666667
        'DOLocationID': '170',
        'PULocationID': '65',
        'trip_distance': 6.54,
    },
]


@custom
def predict(
    model_settings: Dict[str, Tuple[Booster, DictVectorizer]],
    **kwargs,
) -> List[float]:
    inputs: List[Dict[str, Union[float, int]]] = kwargs.get('inputs', DEFAULT_INPUTS)
    inputs = combine_features(inputs)

    DOLocationID = kwargs.get('DOLocationID')
    PULocationID = kwargs.get('PULocationID')
    trip_distance = kwargs.get('trip_distance')

    if DOLocationID is not None or PULocationID is not None or trip_distance is not None:
        inputs = [
            {
                'DOLocationID': DOLocationID,
                'PULocationID': PULocationID,
                'trip_distance': trip_distance,
            },
        ]
    
    model, vectorizer = model_settings['xgboost']
    vectors = vectorizer.transform(inputs)

    predictions = model.predict(build_data(vectors))

    for idx, input_feature in enumerate(inputs):
        print(f'Prediction of duration using these features: {predictions[idx]}')
        for key, value in inputs[idx].items():
            print(f'\t{key}: {value}')

    return predictions.tolist()

```
</details>


### 3.4.5 Inference notebook
- Run the `GDP`-block of the pipeline
- Run the `inference`-block of the pipeline

Results:
```
Prediction of duration using these features: 12.563854217529297
	DOLocationID: 239
	PULocationID: 236
	trip_distance: 1.98
	PU_DO: 236_239

Prediction of duration using these features: 23.28973960876465
	DOLocationID: 170
	PULocationID: 65
	trip_distance: 6.54
	PU_DO: 65_170
```

### 3.4.6 Interactions & 3.4.7 Interactions run
- Click the icon on the bottom of the right panel in the Mage-UI to open the `Interactions`-Menu
- `Interactions` allows us to give data to the pipeline via some API without the need of code
- The `inference`-block is interacted with here. Some parameters can be passed that are then used in the input data for the model

Interactions can be accessed via a tab on top of the block it is interacting with
![3_4_6_interactions2](imgs/3_4_6_interactions2.png)

<b><u>Set Variables:</u></b>
- `DOLocationID`
  - **Label**: DOLocationID
  - **Description**: Sample values: 239, 170
  - **Valid data types**: Integer
  - **Input**: checkbox
- `PULocationID`
  - **Label**: PULocationID
  - **Description**: Sample values: 236, 65
  - **Valid data types**: Integer
  - **Input**: options
    - *Type*: Dropdown menu
    - *Options for dropdown menu*: 
      - Label:  236, Value:  236
      - Label:   65, Value:   65
      - Label:    2, Value:    2
      - Label: 2000, Value: 2000

- `trip_distance`
  - **Label**: trip_distance
  - **Description**: Sample values: 1.98, 6.54
  - **Valid data types**: Float
  - **Input**: text
    - *Type*: Text field
      - Numbers only

### 3.4.8 API
- An API trigger has to be set up, that exposes a `infernce`-block's function (which calls the model)
- <u><b>Create Trigger</b></u>:
  - **Trigger type**: `API`
  - **Trigger name**: `Real-time predictions`
  - **Trigger description**: `Online inference endpoint`
  - Tick the **Endpoint** Slider above Payload section
  - *Save* the trigger, then *enbable* the trigger
- Click `Save trigger in code` (relevant for version control)
- Copy `API endpoint` URL and extract the number at the end (example: `ef46ef6d89f44853b3d31ef771991bc5`) and insert it into the curl command from the Markdown-block with inference instructions
  - `--header 'Authorization: Bearer ef46ef6d89f44853b3d31ef771991bc5'`
  
**Final command for API-call**
```bash
curl --location 'http://localhost:6789/api/runs' \
--header 'Authorization: Bearer ef46ef6d89f44853b3d31ef771991bc5' \
--header 'Content-Type: application/json' \
--header 'Cookie: lng=en' \
--data '{
    "run": {
        "pipeline_uuid": "predict",
        "block_uuid": "inference",
        "variables": {
            "inputs": [
                {
                    "DOLocationID": "239",
                    "PULocationID": "236",
                    "trip_distance": 1.98
                },
                {
                    "DOLocationID": "170",
                    "PULocationID": "65",
                    "trip_distance": 6.54
                }
            ]
        }
    }
}'
```

- Open the console in Mage-UI and paste the command to obtain prediction

```python
{"run": {"output": [12.563854217529297, 23.28973960876465]}}
```
- Check the `runs` of the `Real-time predictions` trigger. You will see the triggered executions of the pipeline after the curl-command is used to provide input-data 


<a id="5-deploying"></a>
## 3.5 Deploying: Running operations in production

### 3.5.1 Setup email

- Open the `deploying_to_production` pipeline and execute the `permissions`-block to generate IAM user and required permissions

<details>
<summary><b>permissions.py</b></summary>

```python
from mlops.utils.deploy.aws import (
    IAM_USER_NAME, 
    POLICY_NAME_TERRAFORM_APPLY_DEPLOY_MAGE,
    POLICY_NAME_TERRAFORM_DESTROY_DELETE_RESOURCES,
    TERRAFORM_APPLY_URL,
    TERRAFORM_DESTROY_URL,
    attach_policy_to_user,
    create_access_key_for_user,
    create_policy,
    create_user,
    reset,
    save_credentials_to_file,
)

if 'custom' not in globals():
    from mage_ai.data_preparation.decorators import custom


@custom
def setup(*args, **kwargs):
    reset(IAM_USER_NAME)

    # Create IAM Policies
    terraform_apply_policy_arn = create_policy(
        POLICY_NAME_TERRAFORM_APPLY_DEPLOY_MAGE, TERRAFORM_APPLY_URL
    )
    terraform_destroy_policy_arn = create_policy(
        POLICY_NAME_TERRAFORM_DESTROY_DELETE_RESOURCES, TERRAFORM_DESTROY_URL
    )

    # Create the user MageDeployer
    create_user(IAM_USER_NAME)

    # Attach policies to the user MageDeployer
    attach_policy_to_user(IAM_USER_NAME, terraform_apply_policy_arn)
    attach_policy_to_user(IAM_USER_NAME, terraform_destroy_policy_arn)

    # Create access key
    access_key, secret_key = create_access_key_for_user(IAM_USER_NAME)
    save_credentials_to_file(IAM_USER_NAME, access_key, secret_key)

```
</details>

- Run the `infrastructure_setup`-block to download and parameterize the configuration for deployment

<details>
<summary><b>infrastructure_setup.py</b></summary>

```python
import os

from mlops.utils.deploy.terraform.env_vars import set_environment_variables
from mlops.utils.deploy.terraform.setup import download_terraform_configurations, setup_configurations

if 'custom' not in globals():
    from mage_ai.data_preparation.decorators import custom


@custom
def setup(*args, **kwargs):
    """
    Downloads the base configurations for Terraform maintained and provided by Mage
    https://github.com/mage-ai/mage-ai-terraform-templates
    """
    download_terraform_configurations()

    """
    1. Updates variables in the Terraform variables file.
    2. Adds variables into the main.tf template env_vars.
    3. Adds environment variables to env_vars.json.

    prevent_destroy_ecr:
        True
    project_name:
        "mlops"
    """
    setup_configurations(
        prevent_destroy_ecr=kwargs.get('prevent_destroy_ecr'),
        project_name=kwargs.get('project_name'),
    )

    """
    Use the current environment variables as the environment variables in production.
    Change this if you want different values.
    In a real world environment, we’d have different values but this is here for 
    demonstration purposes and for convenience.
    """
    set_environment_variables(
        password=kwargs.get('password', os.getenv('POSTGRES_PASSWORD')),
        username=kwargs.get('username', os.getenv('POSTGRES_USER')),
        smtp_email=kwargs.get('smtp_email', os.getenv('SMTP_EMAIL')),
        smtp_password=kwargs.get('smtp_password', os.getenv('SMTP_PASSWORD')),
    )
```
</details>

### 3.5.2 Deploying： Deploy
- Run `teardown_deployed_resources`
  - Does nothing and exits if there are no resources deployed yet
  - Does destroy the deployment resources if they exist

<details>
<summary><b>teardown_deployed_resources.py</b></summary>

```python
from mlops.utils.deploy.terraform.cli import terraform_destroy

if 'custom' not in globals():
    from mage_ai.data_preparation.decorators import custom

@custom
def transform_custom(*args, **kwargs):
    if kwargs.get('destroy'):
        terraform_destroy()
    else:
        print('Skipping Terraform destroy...')
```
</details>

- Run `deploy`-block to create the specified resources with `terraform apply`
- When the `apply`-process is complet, Mage will run in the cloud but in a new state 

<details>
<summary><b>deploy.py</b></summary>

```python
from mlops.utils.deploy.aws import update_boto3_client
from mlops.utils.deploy.terraform.cli import terraform_apply

if 'custom' not in globals():
    from mage_ai.data_preparation.decorators import custom


@custom
def deploy(*args, **kwargs):
    update_boto3_client()
    terraform_apply()
```
</details>

- After the process is done you will get an URL like this, where you can access Mage in AWS:
```bash
http://mlops-production-alb-489725626.us-west-2.elb.amazonaws.com/
```

### 3.5.3 CI⧸CD

> Text is taken from the `Markdown`-blocks from the `deploying_to_production`-pipeline

**Continuous Integration and Continuous Deployment (CI/CD)**

This allows developers to automate the process of testing and deploying code.
In this section, we will create a CI/CD pipeline that will build and
deploy our application to AWS Elastic Container Service (ECS).

<br />

---

<br />

**GitHub Actions YAML configurations**

If you deployed Mage using the Terraform templates provided by Mage,
a GitHub Action workflow YAML file will already be created. 
Skip this section and go to the **Create IAM policy** section to continue.

If you don’t have the GitHub Actions workflow YAML,
follow this [detailed guide](https://docs.mage.ai/production/ci-cd/local-cloud/github-actions)
to create a CI/CD pipeline for your application.

> Note: After running `terraform apply`, a GitHub Actions workflow YAML file should be created
> in the `mage-ops/.github/workflows` directory.
> This file will contain the CI/CD pipeline configuration.

Enter the following environment variables in the GitHub Actions YAML file that match your
infrastructure:

```yaml
env:
  AWS_REGION: us-west-2
  ECR_REPOSITORY: ...
  ECS_CLUSTER: ...
  ECS_SERVICE: ...
  ECS_TASK_DEFINITION: ...
  CONTAINER_NAME: ...
```

<br />

---

<br />

**Create IAM policy**

1. Policy name: **ContinuousIntegrationContinuousDeployment**

2. Permissions using the JSON editor:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "ecr:BatchCheckLayerAvailability",
        "ecr:CompleteLayerUpload",
        "ecr:GetAuthorizationToken",
        "ecr:InitiateLayerUpload",
        "ecr:PutImage",
        "ecr:UploadLayerPart",
        "ecs:DeregisterTaskDefinition",
        "ecs:DescribeClusters",
        "ecs:DescribeServices",
        "ecs:DescribeTaskDefinition",
        "ecs:RegisterTaskDefinition",
        "ecs:UpdateService",
        "iam:PassRole"
      ],
      "Resource": "*"
    }
  ]
}
```

<br />

---

<br />

**Create IAM user**

1. User name: **MageContinuousIntegrationDeployer**

2. Attach policies directly:
   - **ContinuousIntegrationContinuousDeployment**

3. After creating the user, create an access key and secret key.

4. Use case: **Third-party service**


<details>
<summary><b>ci_and_cd.py</b></summary>

```python
from mlops.utils.deploy.aws import (
    IAM_USER_NAME_CICD,
    POLICY_NAME_GITHUB_ACTIONS_DEPLOY_MAGE,
    GITHUB_ACTIONS_DEPLOY_URL,
    attach_policy_to_user,
    create_access_key_for_user,
    create_policy,
    create_user,
    save_credentials_to_file,
)

if 'custom' not in globals():
    from mage_ai.data_preparation.decorators import custom


@custom
def setup(*args, **kwargs):
    # Create IAM policy ContinuousIntegrationContinuousDeployment
    policy_arn = create_policy(
        POLICY_NAME_GITHUB_ACTIONS_DEPLOY_MAGE, GITHUB_ACTIONS_DEPLOY_URL
    )

    # Create the user MageContinuousIntegrationDeployer
    create_user(IAM_USER_NAME_CICD)

    # Attach policy to the user MageContinuousIntegrationDeployer
    attach_policy_to_user(IAM_USER_NAME_CICD, policy_arn)

    # Create access key
    access_key, secret_key = create_access_key_for_user(IAM_USER_NAME_CICD)
    save_credentials_to_file(IAM_USER_NAME_CICD, access_key, secret_key)
```
</details>

To run a github workflow that updates the Mage instance on AWS you have to execute the following steps:
1. Create Github repository for the code (or fork https://github.com/mage-ai/mlops)
2. (Optional) Create Github access token that includes access to github actions / workflows (if is does not exist yet)
3. See your first workflow fail. Dont worrry, it will work after some changes!
4. Go to `Settings`-Tab of your repo
   - In `Security`-subemnu on the sidebar select `Secrets and variables`, then `Actions`
   - Add `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` of `MageContinuousIntegrationDeployer` to the **Repository secrets**
5. Make a change to the code to commit once again and trigger the github workflow
6. Go to the small icon right beside your current commit-message and wait for the process to finish. This can take some time.

### 3.5.4 Deployed

> TODO: Problem -> Git code not used in deployed ECR container???????


<a id="6-homework"></a>
## 3.6 Homework
- Homework [questions](homework/homework.md)
- Homework [solutions](homework/solution.md)