from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pandas as pd

from Core import withSpinner

from .utils import (
    _defaultFeatureName,
    _getActiveDataset,
    _requireColumns,
    _requireNumericColumns,
    _safeDivide,
    _safeUniqueCount,
    _toDataFrame,
    _toTextSeries,
    _validateThreshold,
)


@withSpinner()
def createFeature(name: str, expression: str | Callable[[pd.DataFrame], Any]) -> pd.DataFrame:
    """Create one feature from a pandas expression or callable."""
    frame = _toDataFrame(_getActiveDataset())
    if not name:
        raise ValueError("name must not be empty")

    try:
        frame[name] = expression(frame) if callable(expression) else frame.eval(expression)
    except Exception as exc:
        raise ValueError(f"Could not create feature '{name}' from expression. Reason: {exc}") from exc

    return frame


@withSpinner()
def binColumn(
    column: str,
    bins: int | list[float],
    labels: list[str] | None = None,
    newColumn: str | None = None,
) -> pd.DataFrame:
    """Bucket a numeric column into bins."""
    frame = _toDataFrame(_getActiveDataset())
    _requireNumericColumns(frame, [column])
    output_column = newColumn or _defaultFeatureName(column, "bin")

    try:
        frame[output_column] = pd.cut(frame[column], bins=bins, labels=labels)
    except Exception as exc:
        raise ValueError(f"Could not bin column '{column}'. Check bins/labels. Reason: {exc}") from exc

    return frame


@withSpinner()
def combineColumns(
    columns: list[str],
    newColumn: str,
    separator: str = "_",
) -> pd.DataFrame:
    """Combine multiple columns into one string feature."""
    frame = _toDataFrame(_getActiveDataset())
    if len(columns) < 2:
        raise ValueError("columns must contain at least two column names")
    _requireColumns(frame, columns)

    frame[newColumn] = frame[columns].fillna("").astype("string").agg(separator.join, axis=1)
    return frame


@withSpinner()
def addTextLength(column: str, newColumn: str | None = None) -> pd.DataFrame:
    """Add character-count feature for a text column."""
    frame = _toDataFrame(_getActiveDataset())
    _requireColumns(frame, [column])
    output_column = newColumn or _defaultFeatureName(column, "charCount")
    frame[output_column] = _toTextSeries(frame[column], column).str.len()
    return frame


@withSpinner()
def addWordCount(column: str, newColumn: str | None = None) -> pd.DataFrame:
    """Add word-count feature for a text column."""
    frame = _toDataFrame(_getActiveDataset())
    _requireColumns(frame, [column])
    output_column = newColumn or _defaultFeatureName(column, "wordCount")
    frame[output_column] = _toTextSeries(frame[column], column).str.split().str.len()
    return frame


@withSpinner()
def addAverageWordLength(column: str, newColumn: str | None = None) -> pd.DataFrame:
    """Add average word length feature for a text column."""
    frame = _toDataFrame(_getActiveDataset())
    _requireColumns(frame, [column])
    output_column = newColumn or _defaultFeatureName(column, "avgWordLength")
    words = _toTextSeries(frame[column], column).str.findall(r"\S+")
    frame[output_column] = words.map(
        lambda values: 0.0 if not values else sum(len(value) for value in values) / len(values)
    )
    return frame


@withSpinner()
def addRegexCount(column: str, pattern: str, newColumn: str) -> pd.DataFrame:
    """Count regex matches in a text column."""
    frame = _toDataFrame(_getActiveDataset())
    _requireColumns(frame, [column])
    if not pattern:
        raise ValueError("pattern must not be empty")
    if not newColumn:
        raise ValueError("newColumn must not be empty")

    try:
        frame[newColumn] = _toTextSeries(frame[column], column).str.count(pattern)
    except Exception as exc:
        raise ValueError(f"Invalid regex pattern '{pattern}'. Reason: {exc}") from exc

    return frame


@withSpinner()
def addTextStats(column: str, prefix: str | None = None) -> pd.DataFrame:
    """Add common text statistics for one text column."""
    frame = _toDataFrame(_getActiveDataset())
    _requireColumns(frame, [column])
    text = _toTextSeries(frame[column], column)
    output_prefix = prefix or column

    words = text.str.findall(r"\S+")
    frame[_defaultFeatureName(output_prefix, "charCount")] = text.str.len()
    frame[_defaultFeatureName(output_prefix, "wordCount")] = words.str.len()
    frame[_defaultFeatureName(output_prefix, "avgWordLength")] = words.map(
        lambda values: 0.0 if not values else sum(len(value) for value in values) / len(values)
    )
    frame[_defaultFeatureName(output_prefix, "digitCount")] = text.str.count(r"\d")
    frame[_defaultFeatureName(output_prefix, "uppercaseCount")] = text.str.count(r"[A-Z]")
    frame[_defaultFeatureName(output_prefix, "punctuationCount")] = text.str.count(r"[^\w\s]")
    frame[_defaultFeatureName(output_prefix, "sentenceCount")] = text.str.count(r"[.!?]+")
    return frame


@withSpinner()
def extractDateParts(column: str, prefix: str | None = None) -> pd.DataFrame:
    """Extract year, month, day, weekday, and hour from a date/datetime column."""
    frame = _toDataFrame(_getActiveDataset())
    _requireColumns(frame, [column])
    output_prefix = prefix or column

    converted = pd.to_datetime(frame[column], errors="coerce")
    if converted.notna().sum() == 0 and frame[column].notna().sum() > 0:
        raise ValueError(f"Column '{column}' could not be parsed as datetime")

    frame[_defaultFeatureName(output_prefix, "year")] = converted.dt.year
    frame[_defaultFeatureName(output_prefix, "month")] = converted.dt.month
    frame[_defaultFeatureName(output_prefix, "day")] = converted.dt.day
    frame[_defaultFeatureName(output_prefix, "weekday")] = converted.dt.weekday
    frame[_defaultFeatureName(output_prefix, "hour")] = converted.dt.hour
    return frame


@withSpinner()
def addInteraction(
    columns: list[str],
    operation: str = "multiply",
    newColumn: str | None = None,
) -> pd.DataFrame:
    """Add a numeric interaction feature across columns."""
    frame = _toDataFrame(_getActiveDataset())
    if len(columns) < 2:
        raise ValueError("columns must contain at least two numeric column names")
    _requireNumericColumns(frame, columns)
    output_column = newColumn or _defaultFeatureName(*columns, operation)

    if operation == "multiply":
        frame[output_column] = frame[columns].prod(axis=1)
    elif operation == "add":
        frame[output_column] = frame[columns].sum(axis=1)
    elif operation == "subtract":
        frame[output_column] = frame[columns[0]]
        for column in columns[1:]:
            frame[output_column] = frame[output_column] - frame[column]
    elif operation == "divide":
        frame[output_column] = frame[columns[0]]
        for column in columns[1:]:
            frame[output_column] = _safeDivide(frame[output_column], frame[column])
    else:
        raise ValueError("operation must be 'multiply', 'add', 'subtract', or 'divide'")

    return frame


@withSpinner()
def addRatio(numerator: str, denominator: str, newColumn: str | None = None) -> pd.DataFrame:
    """Add numerator / denominator as a numeric feature."""
    frame = _toDataFrame(_getActiveDataset())
    _requireNumericColumns(frame, [numerator, denominator])
    output_column = newColumn or _defaultFeatureName(numerator, "per", denominator)
    frame[output_column] = _safeDivide(frame[numerator], frame[denominator])
    return frame


@withSpinner()
def dropLowVariance(threshold: float = 0.0) -> pd.DataFrame:
    """Drop numeric columns with variance less than or equal to threshold."""
    _validateThreshold("threshold", threshold)
    frame = _toDataFrame(_getActiveDataset())
    numeric_columns = frame.select_dtypes(include="number").columns.tolist()
    if not numeric_columns:
        raise ValueError("dropLowVariance needs at least one numeric column")

    variances = frame[numeric_columns].var(numeric_only=True)
    columns_to_drop = variances[variances <= threshold].index.tolist()
    return frame.drop(columns=columns_to_drop)


@withSpinner()
def dropHighCardinality(columns: list[str] | None = None, threshold: int = 1000) -> pd.DataFrame:
    """Drop columns with too many unique values."""
    _validateThreshold("threshold", float(threshold))
    frame = _toDataFrame(_getActiveDataset())
    selected_columns = columns or frame.columns.tolist()
    _requireColumns(frame, selected_columns)

    columns_to_drop = [
        column
        for column in selected_columns
        if _safeUniqueCount(frame[column].dropna()) > threshold
    ]
    return frame.drop(columns=columns_to_drop)
