# 2. Experiment tracking and model management

* [Slides](https://drive.google.com/file/d/1YtkAtOQS3wvY7yts_nosVlXrLQBq5q37/view?usp=sharing)


## 2.1 Experiment tracking intro

### Concepts
- **`ML experiment`**: process of building ML model
- **`Experiment Run`**: trial of ML experiment
- **`Run artifact`**: data associated with experiment run
- **`Experiment Metadata`**

### Experiment tracking?!
- <u>Tracking all relevant information</u> of an ML experiment like: Code, Env., Data, Model, Hyperparameter, Metrics, etc.
- Relevance of Infoirmation depends on experiment

### Why is Experiment tracking important?
- **`Reproducibility`**: Just as with other sciences, Data Scientists want to be able to reproduce results of experiments
- **`Organization`**: Helps to organize the project when working with other people. Even when working alone, a certain amount of organization is usually beneficial for success of a project.
- **`Optimization`**: Automates the optimization of ML models.

### Tracking Experiments in Spreadsheets (is not enough!)
- Often filled manually and is error prone
- Spreadsheets are not a standard format for experiment tracking

### MLflow (a good alternative!)
[MLflow-Documentation](https://mlflow.org/docs/latest/index.html)

- Open Source platform for ML Lifecycle
- Installed as <u>python package with 4 main modules</u>: 
    - `Tracking`: focused on experiment tracking
    - `Models`: standard format for packaging ML models that can be used in downstream tools
    - `Model Registry`: centralized model store, set of APIs, and UI, to collaboratively <u>manage the full lifecycle of an MLflow Model</u>
    - `Projects`

### Using MLflow for experiment tracking
- Organization of experiments into runs to keep track of:
    - `Parameters`: path to dataset, preprocessing parameters, ...
    - `Metrics`: e.g. Accuracy, Precision, Recall from training and validation data, ...
    - `Metadata`: e.g. adding tags to runs, s.t. can be searched for in the future. Saving such information allows comparison between runs.
    - `Artifacts`: Any file that is used/generated with the model and is considered relevant for the run.
    - `Models`: Saving/Logging the model (if wanted). Often saving parameters that lead to the model is sufficient.
- Additionally the following information is also logged:
    - `Source Code`
    - `Version (Git)`: current commit
    - `Start- & End-Time`
    - `Author`

### MLflow Demo
Install **`MLflow`**
```bash
pip install mlflow
```

**Opening the MLflow UI**
```bash
mlflow ui
```
<p align="center">
    <img src="imgs/mlflow_ui.png", style="max-width: 60%; height: auto;">
    <figcaption align="center"><b>Figure 1.</b> The MLflow User Interface</figcaption>
</p>

Create a new experiment by clicking on the plus sign on the upper left and filling out the dialogue window:
<p align="center">
    <img src="imgs/experiment_dialogue.png", style="max-width: 60%; height: auto;">
    <figcaption align="center"><b>Figure 2.</b> Creating a new (empty) experiment</figcaption>
</p>

- When experiments are conducted, the results will show where currently the text "No runs logged" is.
- You can specify which columns to use for logging by checking them when clicking the `columns` button

**Looking at models**
<div style="display: flex; justify-content: center;">
    <img src="imgs/models_list.png" style="max-width: 50%; height: auto;">
    <img src="imgs/models_error.png" style="max-width: 50%; height: auto;">
</div>
On the left you can see an empty list of registered models from the freshly created experiment. On the right you can see an error that is caused by not having one of the listed databased installed as a backend for MLflow.


## 2.2 Getting started with MLflow

This part show how to set up and run experiments with MLflow.

### Create MLflow conda-environment
```sh
conda create -n exp-tracking-env python=3.9 pip
```
The `pip` at the end is sometimes required when working on AWS, when `python3-pip` is not installed on the EC2-Instance. On your home computer you most probably have pip already installed, however it is advised to specify it at the creation of a new environment.

### Installing requiements (from file)
Install all dependencies from [mlflow_requirements.txt](mlflow_requirements.txt) with 
```sh
pip install -r mlflow_requirements.txt
```
<u>List of required packages</u>
```bash
mlflow
jupyter
scikit-learn
pandas
seaborn
hyperopt
xgboost
```

### Running MLflow and opening the UI

#### **`Running MLflow locally`**
When running the UI locally you can proceed as seen before:
```sh
mlflow ui [-parameters]
```

#### **`Accessing UI when running MLflow on AWS`**

1. Start the MLflow UI
    ```sh
    # will show http://127.0.0.1:5000
    mlflow ui [-parameters]
    ```
2. New ssh-connection with port forwarding
    ```sh
    ssh -L 5000:localhost:5000 <aws-ec2-name>
    ```
3. You can now open the MLflow UI on your local machine at: `http://127.0.0.1:5000`

#### **`Explicitely specifying a Database-Backend`**

Choosing a specific database type to save the results of experiments to (here: `sqlite`-database)
```sh
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
When clicking on the `models`-tab in the MLflow UI you should not get an error now.

// TODO

## 2.3 Experiment tracking with MLflow

## 2.4 Model management

## 2.5 Model registry

## 2.6 MLflow in practice

## 2.7 MLflow: benefits, limitations and alternatives

## 2.7 Homework
