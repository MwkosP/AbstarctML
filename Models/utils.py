from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from Data.utils import _getActiveDataset, _toDataFrame


_MODEL_REGISTRY = {
    "logistic_regression": LogisticRegression,
    "linear_regression": LinearRegression,
    "random_forest_classifier": RandomForestClassifier,
    "random_forest_regressor": RandomForestRegressor,
    "decision_tree_classifier": DecisionTreeClassifier,
    "decision_tree_regressor": DecisionTreeRegressor,
    "knn_classifier": KNeighborsClassifier,
}


def _getModelClass(name: str) -> type:
    if name not in _MODEL_REGISTRY:
        available = ", ".join(sorted(_MODEL_REGISTRY))
        raise ValueError(f"Unknown model '{name}'. Available: {available}")

    return _MODEL_REGISTRY[name]


def _listModelNames() -> list[str]:
    return sorted(_MODEL_REGISTRY)


def _prepareActiveDataset(
    target: str,
    features: list[str] | None,
    test_size: float,
    random_state: int,
) -> tuple[Any, Any, Any, Any]:
    frame = _toDataFrame(_getActiveDataset())
    X, y = _splitFeaturesTarget(frame, target, features)
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def _preparePredictionData(features: list[str] | None = None) -> Any:
    frame = _toDataFrame(_getActiveDataset())
    if features is None and "features" in frame.columns:
        return _expandFeatureColumn(frame["features"])

    selected_features = features or frame.columns.tolist()
    return pd.get_dummies(frame[selected_features])


def _splitFeaturesTarget(frame: pd.DataFrame, target: str, features: list[str] | None) -> tuple[Any, Any]:
    if target not in frame.columns:
        raise ValueError(f"Target column '{target}' not found")

    y = frame[target]
    if features is None and "features" in frame.columns:
        return _expandFeatureColumn(frame["features"]), y

    selected_features = features or [column for column in frame.columns if column != target]
    X = pd.get_dummies(frame[selected_features])
    return X, y


def _expandFeatureColumn(series: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(series.tolist(), index=series.index)
