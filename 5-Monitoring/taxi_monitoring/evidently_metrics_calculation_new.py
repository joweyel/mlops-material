# %% [markdown]
# # Baseline model for batch monitoring example (Updated for Evidently 0.7.7+)

# %%
import requests
import datetime
import pandas as pd
import matplotlib.pyplot as plt

from joblib import load, dump
from tqdm import tqdm

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

from evidently.report import Report
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

# %% [markdown]
# **Download data and save to `data` folder**

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

# %%
jan_data["duration_min"] = jan_data["lpep_dropoff_datetime"] - jan_data["lpep_pickup_datetime"]
jan_data["duration_min"] = jan_data["duration_min"].apply(lambda td: float(td.total_seconds()) / 60)

jan_data = jan_data[(jan_data["duration_min"] >= 0) & (jan_data["duration_min"] <= 60)]
jan_data = jan_data[(jan_data["passenger_count"] > 0) & (jan_data["passenger_count"] <= 8)]

# %% [markdown]
# # Feature Selection and Model Training

# %%
target = "duration_min"
num_features = ["passenger_count", "trip_distance", "fare_amount", "total_amount"]
cat_features = ["PULocationID", "DOLocationID"]
features = num_features + cat_features

N = jan_data.shape[0]
M = 30_000
train_data = jan_data[:M]
val_data = jan_data[M:]

X_train, y_train = train_data[features], train_data[target]
X_val, y_val = val_data[features], val_data[target]

model = LinearRegression()
model.fit(X_train, y_train)

train_data["prediction"] = model.predict(X_train)
val_data["prediction"] = model.predict(X_val)

mae_train = mean_absolute_error(train_data[target], train_data["prediction"])
mae_val = mean_absolute_error(val_data[target], val_data["prediction"])
print(f"MAE(train): {mae_train:.2f} min")
print(f"MAE(val): {mae_val:.2f} min")

# %% [markdown]
# # Save Model and Data

# %%
with open("models/lin_reg.bin", "wb") as f_out:
    dump(model, f_out)

val_data.to_parquet("data/reference.parquet")

# %% [markdown]
# # Evidently Report

# %%
report = Report(metrics=[DataDriftPreset(), DataQualityPreset()])
report.run(reference_data=train_data, current_data=val_data)
report.show(mode="inline")

# %%
result = report.as_dict()

print("Metric entries in results:")
for i, metric in enumerate(result["metrics"]):
    print(i, metric["metric"])
print("\n", result["metrics"][0].keys())
print(result["metrics"][0]["metric"])
print(result["metrics"][0]["result"].keys())

drift_score = float(result["metrics"][0]["result"]["drift_share"])
missing_share = float(result["metrics"][1]["result"]["current"]["share_of_missing_values"])

print(f"\nDrift Share: {drift_score:.2f}")
print(f"Share of missing values: {missing_share:.2f}")

# %% [markdown]
# # Evidently Dashboard

# %%
ws = Workspace("workspace")
project = ws.create_project("NYC Taxi Data Quality Project")
project.description = "My project description"
project.save()

regular_report = Report(
    metrics=[DataQualityPreset()],
    timestamp=datetime.datetime(2022, 1, 28)
)

filtered_data = val_data.loc[val_data["lpep_pickup_datetime"].between("2022-01-28", "2022-01-29", inclusive="left")]
regular_report.run(reference_data=None, current_data=filtered_data)
ws.add_report(project.id, regular_report)

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
    metrics=[DataQualityPreset()],
    timestamp=datetime.datetime(2022, 1, 29)
)

filtered_data = val_data.loc[val_data["lpep_pickup_datetime"].between("2022-01-28", "2022-01-30", inclusive="left")]
regular_report.run(reference_data=None, current_data=filtered_data)
ws.add_report(project.id, regular_report)
