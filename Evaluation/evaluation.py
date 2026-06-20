from __future__ import annotations

from collections.abc import Callable
from typing import Any

from Core import withSpinner

from .utils import _inferPredictions, _listMetricNames, _registerMetric, _resolveMetrics, _safeMetricCall


@withSpinner()
def listAvailableMetrics(task: str | None = None) -> list[str] | dict[str, list[str]]:
    """List built-in and registered metrics."""
    metrics = _listMetricNames(task)
    print(metrics)
    return metrics


@withSpinner()
def evaluateModel(
    trained: dict[str, Any] | None = None,
    model: Any | None = None,
    X: Any | None = None,
    yTrue: Any | None = None,
    yPred: Any | None = None,
    task: str = "classification",
    metrics: str | list[str] | Callable[..., Any] | dict[str, Callable[..., Any] | str] | None = None,
) -> dict[str, Any]:
    """Evaluate model predictions with built-in or custom metrics."""
    if yTrue is None:
        yTrue, yPred = _inferPredictions(trained, model, yPred)
    elif yPred is None:
        _, yPred = _inferPredictions(trained, model, yPred)

    if X is None and trained is not None and "XTest" in trained:
        X = trained["XTest"]

    if yTrue is None and task != "clustering":
        raise ValueError("Pass yTrue when evaluating explicit yPred values")

    resolved_metrics = _resolveMetrics(metrics, task)
    return {name: _safeMetricCall(metric, yTrue, yPred, X, name) for name, metric in resolved_metrics.items()}


@withSpinner()
def runMetric(
    yTrue: Any = None,
    yPred: Any = None,
    metric: str | Callable[..., Any] = "accuracy",
    task: str = "classification",
    X: Any | None = None,
) -> Any:
    """Run a single metric against yTrue and yPred."""
    resolved = _resolveMetrics(metric, task)
    name, metric_fn = next(iter(resolved.items()))
    return {name: _safeMetricCall(metric_fn, yTrue, yPred, X, name)}


@withSpinner()
def registerMetric(name: str, metric: Callable[..., Any], task: str = "classification") -> None:
    """Register a custom metric callable."""
    _registerMetric(name, metric, task)
