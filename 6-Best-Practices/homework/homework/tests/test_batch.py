import pandas as pd
from datetime import datetime

from batch2 import prepare_data

def dt(hour, minute, second=0):
    return datetime(2023, 1, 1, hour, minute, second)

def test_prepare_data():
    data = [
        (None, None, dt(1, 1), dt(1, 10)),
        (1, 1, dt(1, 2), dt(1, 10)),
        (1, None, dt(1, 2, 0), dt(1, 2, 59)),
        (3, 4, dt(1, 2, 0), dt(2, 2, 1)),      
    ]

    columns = ["PULocationID", "DOLocationID", "tpep_pickup_datetime", "tpep_dropoff_datetime"]
    df = pd.DataFrame(data, columns=columns)
    
    categorical = ["PULocationID", "DOLocationID"]
    df_test = prepare_data(df, categorical)
    
    target_features = ["PULocationID", "DOLocationID", "duration"]
    df_target = pd.DataFrame([
        ("-1", "-1", 9.0),
        ( "1",  "1", 8.0),
    ], columns=target_features)
    
    assert (df_target == df_test[target_features]).values.sum() == 6
    assert (df_test["PULocationID"] == df_target["PULocationID"]).all()
    assert (df_test["DOLocationID"] == df_target["DOLocationID"]).all()
    assert (df_test["duration"] == df_target["duration"]).all()
    