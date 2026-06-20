from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from Core import withSpinner

from .utils import (
    _detectColumnTypes,
    _describeSummary,
    _encodeColors,
    _getActiveDataset,
    _isListLikeColumn,
    _prepareOutputDir,
    _printDataFrame,
    _printSchema,
    _printWarning,
    _projectFeatureMatrix,
    _safeUniqueCount,
    _saveFigure,
    _selectPlottableColumns,
    _showFigure,
    _toDataFrame,
)


@withSpinner()
def describeData(show: bool = True, sample: int | None = None) -> pd.DataFrame:
    """Print and return mixed-type descriptive stats per column."""
    frame = _toDataFrame(_getActiveDataset())
    total_rows = len(frame)
    if sample is not None and total_rows > sample:
        frame = frame.sample(n=sample, random_state=42)

    rows: list[dict[str, Any]] = []

    for column in frame.columns:
        series = frame[column]
        non_null = series.dropna()
        row: dict[str, Any] = {
            "column": column,
            "dtype": str(series.dtype),
            "count": int(non_null.count()),
            "missingCount": int(series.isna().sum()),
            "uniqueCount": _safeUniqueCount(non_null),
        }

        if pd.api.types.is_numeric_dtype(series):
            row.update(
                {
                    "kind": "numeric",
                    "mean": non_null.mean() if len(non_null) else None,
                    "std": non_null.std() if len(non_null) else None,
                    "min": non_null.min() if len(non_null) else None,
                    "25%": non_null.quantile(0.25) if len(non_null) else None,
                    "50%": non_null.quantile(0.50) if len(non_null) else None,
                    "75%": non_null.quantile(0.75) if len(non_null) else None,
                    "max": non_null.max() if len(non_null) else None,
                }
            )
        elif pd.api.types.is_bool_dtype(series):
            row.update(
                {
                    "kind": "boolean",
                    "trueCount": int(non_null.sum()) if len(non_null) else 0,
                    "falseCount": int((~non_null).sum()) if len(non_null) else 0,
                }
            )
        elif pd.api.types.is_datetime64_any_dtype(series):
            row.update(
                {
                    "kind": "datetime",
                    "min": non_null.min() if len(non_null) else None,
                    "max": non_null.max() if len(non_null) else None,
                }
            )
        elif _isListLikeColumn(non_null):
            list_lengths = non_null.map(len)
            text_lengths = non_null.map(lambda value: sum(len(str(item)) for item in value))
            row.update(
                {
                    "kind": "list",
                    "avgListLength": list_lengths.mean() if len(list_lengths) else None,
                    "minListLength": list_lengths.min() if len(list_lengths) else None,
                    "maxListLength": list_lengths.max() if len(list_lengths) else None,
                    "avgTextLength": text_lengths.mean() if len(text_lengths) else None,
                    "minTextLength": text_lengths.min() if len(text_lengths) else None,
                    "maxTextLength": text_lengths.max() if len(text_lengths) else None,
                }
            )
        else:
            text_series = non_null.astype("string")
            lengths = text_series.str.len()
            mode = text_series.mode(dropna=True)
            row.update(
                {
                    "kind": "text",
                    "avgTextLength": lengths.mean() if len(lengths) else None,
                    "minTextLength": lengths.min() if len(lengths) else None,
                    "maxTextLength": lengths.max() if len(lengths) else None,
                    "topValue": mode.iloc[0] if len(mode) else None,
                    "topValueCount": int((text_series == mode.iloc[0]).sum()) if len(mode) else 0,
                }
            )

        rows.append(row)

    description = pd.DataFrame(rows).set_index("column")
    if show:
        printable = description.reset_index()[["column", "kind", "dtype", "count", "missingCount", "uniqueCount"]]
        printable["summary"] = description.apply(_describeSummary, axis=1).to_list()
        title = f"Data Description (sample {len(frame)} of {total_rows})" if len(frame) != total_rows else "Data Description"
        _printDataFrame(printable, title)

    return description


@withSpinner()
def getMissingness() -> pd.DataFrame:
    """Return null counts and percentages per column."""
    frame = _toDataFrame(_getActiveDataset())
    null_count = frame.isna().sum()
    return pd.DataFrame(
        {
            "missingCount": null_count,
            "missingPercent": null_count / len(frame) * 100 if len(frame) else 0,
        }
    )


@withSpinner()
def getCorrelations() -> pd.DataFrame:
    """Return the numeric-column correlation matrix."""
    frame = _toDataFrame(_getActiveDataset())
    return frame.select_dtypes(include="number").corr()


@withSpinner()
def getOutliers(method: str = "iqr", zThreshold: float = 3.0) -> pd.DataFrame:
    """Return outlier counts per numeric column using IQR or z-score."""
    frame = _toDataFrame(_getActiveDataset())
    numeric_frame = frame.select_dtypes(include="number")
    rows: list[dict[str, Any]] = []

    for column in numeric_frame.columns:
        series = numeric_frame[column].dropna()
        if series.empty:
            rows.append({"column": column, "outlierCount": 0, "outlierPercent": 0.0})
            continue

        if method == "iqr":
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            outliers = series[(series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)]
        elif method == "zscore":
            std = series.std()
            outliers = series.iloc[0:0] if std == 0 else series[((series - series.mean()) / std).abs() > zThreshold]
        else:
            raise ValueError("method must be 'iqr' or 'zscore'")

        rows.append(
            {
                "column": column,
                "outlierCount": int(len(outliers)),
                "outlierPercent": float(len(outliers) / len(series) * 100),
            }
        )

    return pd.DataFrame(rows)


@withSpinner()
def viewHead(n: int = 5) -> pd.DataFrame:
    """Print and return the first n rows."""
    frame = _toDataFrame(_getActiveDataset())
    head = frame.head(n)
    _printDataFrame(head, f"Head ({n})")
    return head


@withSpinner()
def viewSample(n: int = 5) -> pd.DataFrame:
    """Print and return a random sample of n rows."""
    frame = _toDataFrame(_getActiveDataset())
    sample = frame.sample(n=min(n, len(frame))) if len(frame) else frame
    _printDataFrame(sample, f"Sample ({n})")
    return sample


@withSpinner()
def viewSchema() -> dict[str, Any]:
    """Print and return dtypes, column names, shape, and inferred column groups."""
    frame = _toDataFrame(_getActiveDataset())
    schema = {
        "shape": frame.shape,
        "columns": frame.columns.tolist(),
        "dtypes": {column: str(dtype) for column, dtype in frame.dtypes.items()},
        "columnTypes": _detectColumnTypes(frame),
    }
    _printSchema(schema)
    return schema


@withSpinner()
def plotDistributions(
    columns: list[str] | None = None,
    outputDir: str | Path | None = None,
    show: bool = False,
) -> dict[str, Any]:
    """Plot one distribution per selected plottable column."""
    frame = _toDataFrame(_getActiveDataset())
    selected_columns = columns or _selectPlottableColumns(frame)
    output_dir = _prepareOutputDir(outputDir)
    figures: dict[str, Any] = {}

    for column in selected_columns:
        if column not in frame.columns:
            continue

        fig, ax = plt.subplots()
        if pd.api.types.is_numeric_dtype(frame[column]):
            frame[column].dropna().plot(kind="hist", bins=30, ax=ax, title=column)
        else:
            frame[column].astype("string").value_counts(dropna=False).head(25).plot(kind="bar", ax=ax, title=column)

        figures[column] = _saveFigure(fig, output_dir / f"{column}_distribution.png" if output_dir else None)
        _showFigure(fig, show)

    return figures


@withSpinner()
def plotDataset(
    dimensions: int = 2,
    method: str = "pca",
    colorBy: str | None = None,
    columns: list[str] | None = None,
    sample: int | None = None,
    outputPath: str | Path | None = None,
    show: bool = False,
) -> Any:
    """Plot the active dataset in 2D or 3D using PCA or t-SNE."""
    frame = _toDataFrame(_getActiveDataset())
    if sample is not None and len(frame) > sample:
        frame = frame.sample(n=sample, random_state=42)

    projected = _projectFeatureMatrix(frame, dimensions=dimensions, method=method, columns=columns)
    colors = frame[colorBy] if colorBy is not None and colorBy in frame.columns else None
    fig = plt.figure()

    if dimensions == 2:
        ax = fig.add_subplot(111)
        scatter = ax.scatter(projected["component1"], projected["component2"], c=_encodeColors(colors), cmap="viridis")
        ax.set_xlabel("component1")
        ax.set_ylabel("component2")
    else:
        ax = fig.add_subplot(111, projection="3d")
        scatter = ax.scatter(
            projected["component1"],
            projected["component2"],
            projected["component3"],
            c=_encodeColors(colors),
            cmap="viridis",
        )
        ax.set_xlabel("component1")
        ax.set_ylabel("component2")
        ax.set_zlabel("component3")

    ax.set_title(f"{method.upper()} {dimensions}D dataset view")
    if colors is not None:
        fig.colorbar(scatter, ax=ax)

    figure = _saveFigure(fig, outputPath)
    _showFigure(fig, show)
    return figure


@withSpinner()
def plotCorrelations(outputPath: str | Path | None = None, show: bool = False) -> Any:
    """Plot a numeric correlation heatmap."""
    correlations = getCorrelations()
    if correlations.empty:
        _printWarning("plotCorrelations needs at least two numeric columns. This dataset has no numeric correlation matrix.")
        return None

    fig, ax = plt.subplots()
    image = ax.imshow(correlations, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(correlations.columns)), correlations.columns, rotation=90)
    ax.set_yticks(range(len(correlations.index)), correlations.index)
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    figure = _saveFigure(fig, outputPath)
    _showFigure(fig, show)
    return figure


@withSpinner()
def plotMissingness(outputPath: str | Path | None = None, show: bool = False) -> Any:
    """Plot missing value percentages per column."""
    missingness = getMissingness()
    fig, ax = plt.subplots()
    missingness["missingPercent"].plot(kind="bar", ax=ax, title="Missingness")
    ax.set_ylabel("Missing %")
    fig.tight_layout()
    figure = _saveFigure(fig, outputPath)
    _showFigure(fig, show)
    return figure


@withSpinner()
def plotOutliers(
    columns: list[str] | None = None,
    outputDir: str | Path | None = None,
    show: bool = False,
) -> dict[str, Any]:
    """Plot boxplots for numeric columns."""
    frame = _toDataFrame(_getActiveDataset())
    numeric_columns = frame.select_dtypes(include="number").columns.tolist()
    selected_columns = columns or numeric_columns
    if not selected_columns:
        _printWarning("plotOutliers needs numeric columns. This dataset has no numeric columns to plot.")
        return {}

    output_dir = _prepareOutputDir(outputDir)
    figures: dict[str, Any] = {}

    for column in selected_columns:
        if column not in numeric_columns:
            continue

        fig, ax = plt.subplots()
        frame.boxplot(column=column, ax=ax)
        ax.set_title(column)
        figures[column] = _saveFigure(fig, output_dir / f"{column}_outliers.png" if output_dir else None)
        _showFigure(fig, show)

    return figures


