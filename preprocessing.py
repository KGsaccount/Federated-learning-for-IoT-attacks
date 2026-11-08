"""Shared data loading and cleaning for the IoT attack dataset.

Every script (baselines, client split, Flower clients) uses these functions,
so all of them see exactly the same features and labels.
"""
import numpy as np
import pandas as pd

TARGET = "Attack Name"
# Identifiers and the binary label are dropped: they identify a flow, not its behaviour.
DROP_COLUMNS = ["Flow ID", "Src IP", "Dst IP", "Timestamp", "Label"]


def load_dataset(path):
    """Read a CSV and return (X, y): numeric features and attack names."""
    df = pd.read_csv(path, on_bad_lines="skip", low_memory=False)
    if TARGET not in df.columns:
        raise ValueError(f"{path}: column '{TARGET}' not found")
    y = df[TARGET].astype(str)
    X = df.drop(columns=DROP_COLUMNS + [TARGET], errors="ignore")
    X = X.select_dtypes(include="number")
    # Flow-rate features often contain +/-inf (division by a zero duration).
    X = X.replace([np.inf, -np.inf], np.nan)
    return X, y


def clean_features(X, columns, fill_values):
    """Align X on the training columns and fill missing values."""
    X = X.reindex(columns=columns)
    return X.fillna(fill_values).fillna(0.0)
