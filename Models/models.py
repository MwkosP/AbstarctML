from __future__ import annotations

from typing import Any

from Core import withSpinner

from .utils import _getModelClass, _listModelNames, _prepareActiveDataset, _preparePredictionData


@withSpinner()
def listAvailableModels() -> list[str]:
    """List model names supported by selectModel."""
    models = _listModelNames()
    print(models)
    return models


@withSpinner()
def selectModel(name: str = "logistic_regression", **params: Any) -> Any:
    """Create a supported sklearn-style model."""
    model_class = _getModelClass(name)
    return model_class(**params)


@withSpinner()
def trainModel(
    model: Any | None = None,
    target: str = "target",
    features: list[str] | None = None,
    testSize: float = 0.2,
    randomState: int = 42,
) -> dict[str, Any]:
    """Train a model on the active dataset and return model plus split data."""
    model = model or selectModel()
    X_train, X_test, y_train, y_test = _prepareActiveDataset(target, features, testSize, randomState)
    model.fit(X_train, y_train)
    return {
        "model": model,
        "XTrain": X_train,
        "XTest": X_test,
        "yTrain": y_train,
        "yTest": y_test,
    }


@withSpinner()
def predictModel(model: Any, features: list[str] | None = None) -> Any:
    """Predict with a trained model on the active dataset."""
    X = _preparePredictionData(features)
    return model.predict(X)
