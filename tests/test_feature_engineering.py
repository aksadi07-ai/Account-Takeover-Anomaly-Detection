import pandas as pd


def test_feature_columns_exist():
    df = pd.DataFrame([{
        "event_count": 4,
        "failed_logins": 0,
        "transfer_count": 1,
        "new_device": 0,
        "new_location": 0,
        "transaction_velocity_per_min": 0.1
    }])
    required = {
        "event_count", "failed_logins", "transfer_count",
        "new_device", "new_location", "transaction_velocity_per_min"
    }
    assert required.issubset(df.columns)
