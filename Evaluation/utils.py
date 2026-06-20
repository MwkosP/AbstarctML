from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    balanced_accuracy_score,
    calinski_harabasz_score,
    classification_report,
    completeness_score,
    confusion_matrix,
    davies_bouldin_score,
    f1_score,
    homogeneity_score,
    mean_absolute_error,
    mean_squared_error,
    normalized_mutual_info_score,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    silhouette_score,
    v_measure_score,
)


_METRIC_REGISTRY: dict[str, dict[str, Callable[..., Any]]] = {
    "classification": {
        "accuracy": accuracy_score,
        "balanced_accuracy": balanced_accuracy_score,
        "precision": lambda y_true, y_pred: precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": lambda y_true, y_pred: recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1": lambda y_true, y_pred: f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "roc_auc": roc_auc_score,
        "confusion_matrix": confusion_matrix,
        "classification_report": lambda y_true, y_pred: classification_report(y_true, y_pred, zero_division=0),
    },
    "regression": {
        "mae": mean_absolute_error,
        "mse": mean_squared_error,
        "rmse": lambda y_true, y_pred: float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": r2_score,
    },
    "clustering": {
        "silhouette": silhouette_score,
        "calinski_harabasz": calinski_harabasz_score,
        "davies_bouldin": davies_bouldin_score,
        "adjusted_rand": adjusted_rand_score,
        "normalized_mutual_info": normalized_mutual_info_score,
        "homogeneity": homogeneity_score,
        "completeness": completeness_score,
        "v_measure": v_measure_score,
    },
}

_DEFAULT_METRICS = {
    "classification": ["accuracy", "precision", "recall", "f1"],
    "regression": ["mae", "mse", "rmse", "r2"],
    "clustering": ["silhouette", "calinski_harabasz", "davies_bouldin"],
}

_CLUSTERING_X_METRICS = {"silhouette", "calinski_harabasz", "davies_bouldin"}


def _getMetric(name: str, task: str) -> Callable[..., Any]:
    if task not in _METRIC_REGISTRY:
        raise ValueError("task must be 'classification', 'regression', or 'clustering'")

    if name not in _METRIC_REGISTRY[task]:
        available = ", ".join(sorted(_METRIC_REGISTRY[task]))
        raise ValueError(f"Unknown {task} metric '{name}'. Available: {available}")

    return _METRIC_REGISTRY[task][name]


def _resolveMetrics(metrics: Any, task: str) -> dict[str, Callable[..., Any]]:
    if metrics is None:
        return {name: _getMetric(name, task) for name in _DEFAULT_METRICS[task]}

    if isinstance(metrics, str):
        return {metrics: _getMetric(metrics, task)}

    if callable(metrics):
        return {getattr(metrics, "__name__", "custom_metric"): metrics}

    if isinstance(metrics, dict):
        resolved: dict[str, Callable[..., Any]] = {}
        for name, metric in metrics.items():
            resolved[name] = _getMetric(metric, task) if isinstance(metric, str) else metric
        return resolved

    return {name: _getMetric(name, task) for name in metrics}


def _registerMetric(name: str, metric: Callable[..., Any], task: str) -> None:
    if task not in _METRIC_REGISTRY:
        raise ValueError("task must be 'classification', 'regression', or 'clustering'")

    _METRIC_REGISTRY[task][name] = metric


def _listMetricNames(task: str | None = None) -> list[str] | dict[str, list[str]]:
    if task is None:
        return {task_name: sorted(metrics) for task_name, metrics in _METRIC_REGISTRY.items()}

    if task not in _METRIC_REGISTRY:
        raise ValueError("task must be 'classification', 'regression', 'clustering', or None")

    return sorted(_METRIC_REGISTRY[task])


def _inferPredictions(trained: dict[str, Any] | None, model: Any | None, y_pred: Any | None) -> tuple[Any, Any]:
    if y_pred is not None:
        if trained is not None and "yTest" in trained:
            return trained["yTest"], y_pred
        return None, y_pred

    if trained is None:
        raise ValueError("Pass trained=... or yPred=...")

    model = model or trained.get("model")
    if model is None or "XTest" not in trained or "yTest" not in trained:
        raise ValueError("trained must include model, XTest, and yTest")

    return trained["yTest"], model.predict(trained["XTest"])


def _safeMetricCall(
    metric: Callable[..., Any],
    y_true: Any,
    y_pred: Any,
    X: Any | None = None,
    metric_name: str | None = None,
) -> Any:
    try:
        if metric_name in _CLUSTERING_X_METRICS:
            if X is None:
                return "Metric unavailable: pass X for this clustering metric"
            return metric(X, y_pred)

        return metric(y_true, y_pred)
    except (TypeError, ValueError) as error:
        return f"Metric unavailable: {error}"
