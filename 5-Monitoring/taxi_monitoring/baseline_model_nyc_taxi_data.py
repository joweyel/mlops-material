# %% [markdown]
# # Baseline model for batch monitoring example

# %%
import requests
import datetime
import pandas as pd
import matplotlib.pyplot as plt

from evidently import ColumnMapping
from evidently.report import Report
from evidently.metrics import (
    ColumnDriftMetric, 
    DatasetDriftMetric,
    DatasetMissingValuesMetric
)

from joblib import load, dump
from tqdm import tqdm

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

# %% [markdown]
# **Download data and saving them for the artifacts folder `data`** 

# %%
files = [('green_tripdata_2022-02.parquet', './data'), ('green_tripdata_2022-01.parquet', './data')]

print("Download files:")
for file, path in files:
    url = f"https://d37ci6vzurychx.cloudfront.net/trip-data/{file}"
    resp = requests.get(url, stream=True)
    save_path = f"{path}/{file}"
    with open(save_path, "wb") as handle:
        for data in tqdm(resp.iter_content(),
                        desc=f"{file}",
                        postfix=f"save to {save_path}",
                        total=int(resp.headers["Content-Length"])):
            handle.write(data)

# %%
jan_data = pd.read_parquet("data/green_tripdata_2022-01.parquet")

# %% [markdown]
# Getting insights into the data:

# %%
jan_data.describe()

# %%
jan_data.shape

# %%
# create target (trip duration in minutes)
jan_data["duration_min"] = jan_data["lpep_dropoff_datetime"] - jan_data["lpep_pickup_datetime"]
jan_data["duration_min"] = jan_data["duration_min"].apply(lambda td: float(td.total_seconds()) / 60)

# %%
# Filter out outliers
jan_data = jan_data[(jan_data["duration_min"] >= 0) & (jan_data["duration_min"] <= 60)]
jan_data = jan_data[(jan_data["passenger_count"] > 0) & (jan_data["passenger_count"] <= 8)]

# %%
jan_data.duration_min.hist()

# %% [markdown]
# # Selecting features and training a linear regression model

# %% [markdown]
# Choosing an adequate subset of features to train the model with:

# %%
# Data labeling
target = "duration_min"
num_features = ["passenger_count", "trip_distance", "fare_amount", "total_amount"]
cat_features = ["PULocationID", "DOLocationID"]
features = num_features + cat_features

# %%
N = jan_data.shape[0]
print("Number of rows: ", N)

# %%
M: int = 30_000
train_data = jan_data[:M]
val_data = jan_data[M:]

X_train, y_train = train_data[features], train_data[target]
X_val, y_val = val_data[features], val_data[target]

print(f"Train data is {(len(train_data)/N * 100.0):.1f}% of the overall data")
print(f"Validation data is {(len(val_data)/N * 100.0):.1f}% of the overall data")

# %%
model = LinearRegression()

# %%
model.fit(X_train, y_train)

# %%
train_preds = model.predict(X_train)
train_data["prediction"] = train_preds.copy()

# %%
val_preds = model.predict(X_val)
val_data["prediction"] = val_preds.copy()

# %%
mae_train = mean_absolute_error(train_data["duration_min"], train_data["prediction"])
mae_val = mean_absolute_error(val_data["duration_min"], val_data["prediction"])

print(f"MAE(train): {mae_train:.2f} min")
print(f"MAE(val): {mae_val:.2f} min")

# %% [markdown]
# # Dump model and reference data

# %%
# Dumping (saving) the model and validation dataset (reference data)
with open("models/lin_reg.bin", "wb") as f_out:
    dump(model, f_out)

val_data.to_parquet("data/reference.parquet")

# %% [markdown]
# # Evidently Report

# %%
column_mapping = ColumnMapping(
    target=None, 
    prediction="prediction",
    numerical_features=num_features,
    categorical_features=cat_features
)

# %%
report = Report(
    metrics=[
        ColumnDriftMetric(column_name="prediction"),
        DatasetDriftMetric(),
        DatasetMissingValuesMetric()
    ]
)

# %%
report.run(
    reference_data=train_data,
    current_data=val_data, 
    column_mapping=column_mapping
)

# %%
report.show(mode="inline")

# %% [markdown]
# HTML-Version of the results is used for visualization purposes. To programmatically process data however, python dictionaries are better suited and will be used.

# %%
result = report.as_dict()

# %%
result

# %% [markdown]
# Deriving any value from an evidently report:

# %%
# Results, check em!
print("Metric entries in results:")
for i in range(len(result["metrics"])):
    print(i, result["metrics"][i]["metric"])
print("\n", result["metrics"][0].keys())
print(result["metrics"][0]["metric"])
print(result["metrics"][0]["result"].keys())

# %%
# prediction drift
float(result["metrics"][0]["result"]["drift_score"])

# %%
# number of drifted columns
float(result["metrics"][1]["result"]["number_of_drifted_columns"])

# %%
# Share of missing values
float(result["metrics"][2]["result"]["current"]["share_of_missing_values"])

# %% [markdown]
# # Evidently Dashboard

# %%
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
from evidently.ui.workspace import Workspace
from evidently.ui.dashboards import (
    DashboardPanelCounter, 
    DashboardPanelPlot,
    CounterAgg, 
    PanelValue, 
    PlotType, 
    ReportFilter
)
from evidently.renderers.html_widgets import WidgetSize

# %%
ws = Workspace("workspace")

# %%
project = ws.create_project("NYC Taxi Data Quality Project")
project.description = "My project description"
project.save()

# %%
regular_report = Report(
    metrics=[
        DataQualityPreset()
    ],
    timestamp=datetime.datetime(2022, 1, 28)
)

regular_report.run(
    reference_data=None,
    current_data=val_data.loc[val_data["lpep_pickup_datetime"].between("2022-01-28", "2022-01-29", inclusive="left")], # select 1 data in the data
    column_mapping=column_mapping
)

regular_report

# %%
# Adding report to workspace
ws.add_report(project.id, regular_report)

# %%
#configure the dashboard
project.dashboard.add_panel(
    DashboardPanelCounter(
        filter=ReportFilter(metadata_values={}, tag_values=[]),
        agg=CounterAgg.NONE,
        title="NYC taxi data dashboard"
    )
)

project.dashboard.add_panel(
    DashboardPanelPlot(
        filter=ReportFilter(metadata_values={}, tag_values=[]),
        title="Inference Count",
        values=[
            PanelValue(
                metric_id="DatasetSummaryMetric",
                field_path="current.number_of_rows",
                legend="count"
            ),
        ],
        plot_type=PlotType.BAR,
        size=WidgetSize.HALF,
    ),
)

project.dashboard.add_panel(
    DashboardPanelPlot(
        filter=ReportFilter(metadata_values={}, tag_values=[]),
        title="Number of Missing Values",
        values=[
            PanelValue(
                metric_id="DatasetSummaryMetric",
                field_path="current.number_of_missing_values",
                legend="count"
            ),
        ],
        plot_type=PlotType.LINE,
        size=WidgetSize.HALF,
    ),
)

project.save()

# %%
regular_report = Report(
    metrics=[
        DataQualityPreset()
    ],
    timestamp=datetime.datetime(2022, 1, 29)
)

regular_report.run(
    reference_data=None,
    current_data=val_data.loc[val_data["lpep_pickup_datetime"].between("2022-01-28", "2022-01-30", inclusive="left")],
    column_mapping=column_mapping
)

regular_report

# %% [markdown]
# Add report to Workspace:

# %%
ws.add_report(project.id, regular_report)


