from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

from Core import withSpinner

from .utils import _getActiveDataset, _getScaler, _toDataFrame


@withSpinner()
def scaleData(method: str = "standard", columns: list[str] | None = None) -> pd.DataFrame:
    """Scale active dataset numeric columns using standard, minmax, or robust scaling."""
    frame = _toDataFrame(_getActiveDataset())
    selected_columns = columns or frame.select_dtypes(include="number").columns.tolist()
    if not selected_columns:
        return frame

    scaler = _getScaler(method)
    frame[selected_columns] = scaler.fit_transform(frame[selected_columns])
    return frame


@withSpinner()
def encodeData(method: str = "onehot", columns: list[str] | None = None) -> pd.DataFrame:
    """Encode active dataset feature columns using onehot, label, or ordinal encoding."""
    frame = _toDataFrame(_getActiveDataset())
    selected_columns = columns or frame.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    if not selected_columns:
        return frame

    if method == "onehot":
        return pd.get_dummies(frame, columns=selected_columns)

    if method == "label":
        for column in selected_columns:
            encoder = LabelEncoder()
            frame[column] = encoder.fit_transform(frame[column].astype("string"))
        return frame

    if method == "ordinal":
        encoder = OrdinalEncoder()
        frame[selected_columns] = encoder.fit_transform(frame[selected_columns].astype("string"))
        return frame

    raise ValueError("method must be 'onehot', 'label', or 'ordinal'")


@withSpinner()
def imputeMissing(
    method: str = "mean",
    columns: list[str] | None = None,
    value: Any | None = None,
) -> pd.DataFrame:
    """Fill active dataset missing values using mean, median, mode, or constant."""
    frame = _toDataFrame(_getActiveDataset())
    selected_columns = columns or frame.columns.tolist()

    for column in selected_columns:
        if method == "mean":
            fill_value = frame[column].mean()
        elif method == "median":
            fill_value = frame[column].median()
        elif method == "mode":
            try:
                mode = frame[column].mode(dropna=True)
            except ValueError:
                mode = frame[column].astype("string").mode(dropna=True)
            fill_value = mode.iloc[0] if len(mode) else value
        elif method == "constant":
            fill_value = value
        else:
            raise ValueError("method must be 'mean', 'median', 'mode', or 'constant'")

        frame[column] = frame[column].fillna(fill_value)

    return frame


@withSpinner()
def dropMissing(
    columns: list[str] | None = None,
    threshold: float | None = None,
) -> pd.DataFrame:
    """Drop active dataset missing rows, or columns over a missingness threshold."""
    frame = _toDataFrame(_getActiveDataset())
    if threshold is not None:
        missing_percent = frame.isna().mean()
        return frame.drop(columns=missing_percent[missing_percent > threshold].index.tolist())

    return frame.dropna(subset=columns)


@withSpinner()
def encodeTarget(y: Any, method: str = "label") -> pd.Series:
    """Encode target values separately from feature data."""
    series = pd.Series(y)
    if method != "label":
        raise ValueError("method must be 'label'")

    encoder = LabelEncoder()
    return pd.Series(encoder.fit_transform(series.astype("string")), name=series.name)
