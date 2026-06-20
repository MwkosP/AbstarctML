from __future__ import annotations

import os
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASETS_DIR = PROJECT_ROOT / "Datasets"
HUGGINGFACE_CACHE_DIR = DATASETS_DIR / "_huggingface_cache"
ENV_FILE = PROJECT_ROOT / ".env"

os.environ["HF_HOME"] = str(HUGGINGFACE_CACHE_DIR / "home")
os.environ["HF_HUB_CACHE"] = str(HUGGINGFACE_CACHE_DIR / "hub")
os.environ["HF_DATASETS_CACHE"] = str(HUGGINGFACE_CACHE_DIR / "datasets")

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
import matplotlib.pyplot as plt
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from sklearn import datasets as sklearn_datasets
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer


_ACTIVE_DATASET_NAME: str | None = None
_ACTIVE_DATASET_SOURCE: str | None = None
_ACTIVE_DATASET_CONFIG_NAME: str | None = None
_ACTIVE_DATASET_SPLIT: str | None = None
_ACTIVE_DATASET_OUTPUT_PATH: str | Path | None = None
_ACTIVE_DATASET_SAVE_AS: str | None = None
_ACTIVE_DATASET: Any | None = None
_CONSOLE = Console()

_SKLEARN_LOADERS = {
    "iris": sklearn_datasets.load_iris,
    "wine": sklearn_datasets.load_wine,
    "breast_cancer": sklearn_datasets.load_breast_cancer,
    "digits": sklearn_datasets.load_digits,
    "diabetes": sklearn_datasets.load_diabetes,
    "linnerud": sklearn_datasets.load_linnerud,
    "california_housing": sklearn_datasets.fetch_california_housing,
}


def _requireActiveDatasetName(name: str | None) -> str:
    if name is not None:
        return name

    if _ACTIVE_DATASET_NAME is None:
        raise ValueError("No active dataset set. Pass a name or call setActiveDataset().")

    return _ACTIVE_DATASET_NAME


def _getActiveDatasetConfig(
    default_source: str,
) -> tuple[str, str | None, str | None, str | Path | None, str | None]:
    return (
        _ACTIVE_DATASET_SOURCE or default_source,
        _ACTIVE_DATASET_CONFIG_NAME,
        _ACTIVE_DATASET_SPLIT,
        _ACTIVE_DATASET_OUTPUT_PATH,
        _ACTIVE_DATASET_SAVE_AS,
    )


def _getActiveDataset() -> Any:
    if _ACTIVE_DATASET is None:
        raise ValueError("No active dataset set. Call setActiveDataset() before using Data EDA functions.")

    return _ACTIVE_DATASET


def _getMatchingActiveDataset(
    name: str,
    source: str,
    config_name: str | None,
    split: str | None,
    output_path: str | Path | None,
    save_as: str | None,
    force: bool,
) -> Any | None:
    if (
        _ACTIVE_DATASET_NAME == name
        and _ACTIVE_DATASET_SOURCE == source
        and _ACTIVE_DATASET_CONFIG_NAME == config_name
        and _ACTIVE_DATASET_SPLIT == split
        and _ACTIVE_DATASET_OUTPUT_PATH == output_path
        and _ACTIVE_DATASET_SAVE_AS == save_as
        and _ACTIVE_DATASET is not None
        and not force
    ):
        return _ACTIVE_DATASET

    return None


def _loadLocalDataset(name: str, output_path: str | Path | None = None) -> Any | None:
    dataset_path = _resolveOutputPath(output_path) / name
    if not dataset_path.exists():
        return None

    if dataset_path.is_dir() and (dataset_path / "data").exists():
        return load_from_disk(str(dataset_path / "data"))

    if dataset_path.is_dir() and (dataset_path / "data.parquet").exists():
        return pd.read_parquet(dataset_path / "data.parquet")

    if dataset_path.suffix == ".parquet":
        return pd.read_parquet(dataset_path)

    if dataset_path.suffix == ".csv":
        return pd.read_csv(dataset_path)

    if dataset_path.suffix == ".json":
        return pd.read_json(dataset_path)

    return None


def _createDatasetSubset(
    name: str,
    save_as: str,
    sample: int,
    output_path: str | Path | None = None,
    split: str | None = None,
    columns: list[str] | None = None,
    stratify_by: str | list[str] | None = None,
    balance: bool = False,
    random_state: int = 42,
) -> tuple[pd.DataFrame, Path]:
    if sample <= 0:
        raise ValueError("sample must be greater than 0")

    dataset_path = _resolveOutputPath(output_path) / save_as
    existing_subset_path = dataset_path / "data.parquet"
    if existing_subset_path.exists():
        return pd.read_parquet(existing_subset_path), dataset_path

    data = _loadLocalDataset(name, output_path)
    if data is None:
        raise ValueError(f"Dataset '{name}' was not found under {_resolveOutputPath(output_path)}")

    data = _selectDatasetSplit(data, split)

    if isinstance(data, (Dataset, DatasetDict)):
        subset = _createHuggingfaceSubset(data, sample, columns, stratify_by, balance, random_state)
        dataset_path.mkdir(parents=True, exist_ok=True)
        subset.to_parquet(dataset_path / "data.parquet", index=False)
        return subset, dataset_path

    frame = _toDataFrame(data)
    frame = _addDerivedSubsetColumns(frame)
    missing_columns = [column for column in columns or [] if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"Columns not found: {missing_columns}")

    stratify_columns = _normalizeColumns(stratify_by)
    missing_stratify_columns = [column for column in stratify_columns if column not in frame.columns]
    if missing_stratify_columns:
        raise ValueError(f"stratifyBy columns not found: {missing_stratify_columns}")

    if stratify_columns and balance:
        subset = _balancedStratifiedSample(frame, stratify_columns, sample, random_state)
    elif stratify_columns:
        subset = _stratifiedSample(frame, stratify_columns, sample, random_state)
    else:
        subset = frame.sample(n=min(sample, len(frame)), random_state=random_state)

    if columns is not None:
        subset = subset[columns]

    dataset_path.mkdir(parents=True, exist_ok=True)
    subset.to_parquet(dataset_path / "data.parquet", index=False)
    return subset, dataset_path


def _normalizeColumns(columns: str | list[str] | None) -> list[str]:
    if columns is None:
        return []
    if isinstance(columns, str):
        return [columns]
    return columns


def _selectDatasetSplit(data: Any, split: str | None) -> Any:
    if split is None:
        return data
    if isinstance(data, DatasetDict):
        if split not in data:
            raise ValueError(f"Split '{split}' was not found. Available splits: {list(data.keys())}")
        return data[split]
    return data


def _createHuggingfaceSubset(
    data: Dataset | DatasetDict,
    sample: int,
    columns: list[str] | None,
    stratify_by: str | list[str] | None,
    balance: bool,
    random_state: int,
) -> pd.DataFrame:
    stratify_columns = _normalizeColumns(stratify_by)
    available_columns = _getHuggingfaceColumns(data)
    requested_columns = columns or sorted(available_columns)
    if columns is None:
        requested_columns = sorted(set(requested_columns) | set(stratify_columns))

    missing_columns = [column for column in requested_columns if column not in available_columns and column != "label"]
    if missing_columns:
        raise ValueError(f"Columns not found: {missing_columns}")

    missing_stratify_columns = [
        column
        for column in stratify_columns
        if column not in available_columns and not (column == "label" and "model" in available_columns)
    ]
    if missing_stratify_columns:
        raise ValueError(f"stratifyBy columns not found: {missing_stratify_columns}")

    count_columns = sorted({column for column in stratify_columns if column != "label"} | {"model"})
    counts = _countHuggingfaceGroups(data, stratify_columns, count_columns)
    targets = _balancedTargets(counts, stratify_columns, sample) if balance else _proportionalTargets(counts, sample)
    rows = _reservoirSampleHuggingfaceRows(data, targets, requested_columns, stratify_columns, random_state)
    return pd.DataFrame(rows).sample(frac=1, random_state=random_state).reset_index(drop=True)


def _getHuggingfaceColumns(data: Dataset | DatasetDict) -> set[str]:
    if isinstance(data, Dataset):
        return set(data.column_names)

    columns: set[str] = {"__split"}
    for dataset in data.values():
        columns.update(dataset.column_names)
    return columns


def _iterHuggingfaceRows(
    data: Dataset | DatasetDict,
    columns: list[str] | None = None,
):
    if isinstance(data, Dataset):
        selected = [column for column in columns or data.column_names if column in data.column_names]
        dataset = data.select_columns(selected) if selected else data
        yield from dataset
        return

    for split_name, dataset in data.items():
        selected = [column for column in columns or dataset.column_names if column in dataset.column_names]
        selected_dataset = dataset.select_columns(selected) if selected else dataset
        for row in selected_dataset:
            row = dict(row)
            row["__split"] = split_name
            yield row


def _countHuggingfaceGroups(
    data: Dataset | DatasetDict,
    stratify_columns: list[str],
    count_columns: list[str],
) -> Counter[tuple[Any, ...]]:
    counts: Counter[tuple[Any, ...]] = Counter()
    for row in _iterHuggingfaceRows(data, count_columns):
        group_key = _rowGroupKey(row, stratify_columns)
        counts[group_key] += 1
    return counts


def _rowGroupKey(row: dict[str, Any], columns: list[str]) -> tuple[Any, ...]:
    if not columns:
        return ("__all__",)
    return tuple(_rowValue(row, column) for column in columns)


def _rowValue(row: dict[str, Any], column: str) -> Any:
    if column == "label" and "label" not in row and "model" in row:
        return "human" if row["model"] == "human" else "ai"
    return row.get(column)


def _balancedTargets(
    counts: Counter[tuple[Any, ...]],
    columns: list[str],
    sample: int,
) -> dict[tuple[Any, ...], int]:
    if not counts:
        return {}
    if not columns:
        total = sum(counts.values())
        return {("__all__",): min(sample, total)}

    targets: dict[tuple[Any, ...], int] = {}
    shortages: list[str] = []
    _allocateBalancedTargets(counts, columns, (), sample, targets, shortages)
    if shortages:
        raise ValueError("Not enough rows to create a balanced subset. " + ", ".join(shortages))
    return targets


def _allocateBalancedTargets(
    counts: Counter[tuple[Any, ...]],
    columns: list[str],
    prefix: tuple[Any, ...],
    sample: int,
    targets: dict[tuple[Any, ...], int],
    shortages: list[str],
) -> None:
    level = len(prefix)
    if level == len(columns):
        available = counts[prefix]
        if available < sample:
            shortages.append(f"{prefix}: needed {sample}, available {available}")
            return
        targets[prefix] = sample
        return

    child_values = sorted(
        {group_key[level] for group_key in counts if group_key[:level] == prefix},
        key=lambda value: str(value),
    )
    base_sample = sample // len(child_values)
    remainder = sample % len(child_values)

    for index, child_value in enumerate(child_values):
        child_prefix = prefix + (child_value,)
        child_sample = base_sample + (1 if index < remainder else 0)
        _allocateBalancedTargets(counts, columns, child_prefix, child_sample, targets, shortages)


def _proportionalTargets(counts: Counter[tuple[Any, ...]], sample: int) -> dict[tuple[Any, ...], int]:
    total = sum(counts.values())
    if total <= sample:
        return dict(counts)

    raw_targets = {
        group_key: sample * count / total
        for group_key, count in counts.items()
    }
    targets = {
        group_key: min(int(target), counts[group_key])
        for group_key, target in raw_targets.items()
    }
    remaining = sample - sum(targets.values())
    ranked_groups = sorted(
        raw_targets,
        key=lambda group_key: (raw_targets[group_key] - targets[group_key], counts[group_key]),
        reverse=True,
    )
    while remaining > 0:
        changed = False
        for group_key in ranked_groups:
            if targets[group_key] < counts[group_key]:
                targets[group_key] += 1
                remaining -= 1
                changed = True
                if remaining == 0:
                    break
        if not changed:
            break
    return {group_key: target for group_key, target in targets.items() if target > 0}


def _reservoirSampleHuggingfaceRows(
    data: Dataset | DatasetDict,
    targets: dict[tuple[Any, ...], int],
    columns: list[str],
    stratify_columns: list[str],
    random_state: int,
) -> list[dict[str, Any]]:
    rng = random.Random(random_state)
    seen: defaultdict[tuple[Any, ...], int] = defaultdict(int)
    reservoirs: dict[tuple[Any, ...], list[dict[str, Any]]] = {group_key: [] for group_key in targets}
    read_columns = sorted(set(columns + [column for column in stratify_columns if column != "label"] + ["model"]))

    for row in _iterHuggingfaceRows(data, read_columns):
        group_key = _rowGroupKey(row, stratify_columns)
        target = targets.get(group_key)
        if target is None:
            continue

        seen[group_key] += 1
        output_row = _selectOutputRow(row, columns)
        reservoir = reservoirs[group_key]
        if len(reservoir) < target:
            reservoir.append(output_row)
            continue

        replacement_index = rng.randint(1, seen[group_key])
        if replacement_index <= target:
            reservoir[replacement_index - 1] = output_row

    return [row for reservoir in reservoirs.values() for row in reservoir]


def _selectOutputRow(row: dict[str, Any], columns: list[str]) -> dict[str, Any]:
    output = {}
    for column in columns:
        if column == "label":
            output[column] = _rowValue(row, column)
        elif column in row:
            output[column] = row[column]
    return output


def _addDerivedSubsetColumns(frame: pd.DataFrame) -> pd.DataFrame:
    if "label" not in frame.columns and "model" in frame.columns:
        frame = frame.copy()
        frame["label"] = frame["model"].map(lambda value: "human" if value == "human" else "ai")
    return frame


def _balancedStratifiedSample(
    frame: pd.DataFrame,
    columns: list[str],
    sample: int,
    random_state: int,
) -> pd.DataFrame:
    return _balancedGroupSample(frame, columns, sample, random_state).sample(frac=1, random_state=random_state)


def _balancedGroupSample(
    frame: pd.DataFrame,
    columns: list[str],
    sample: int,
    random_state: int,
) -> pd.DataFrame:
    if not columns:
        if len(frame) < sample:
            raise ValueError(f"Not enough rows to sample {sample}; available {len(frame)}")
        return frame.sample(n=sample, random_state=random_state)

    groups = list(frame.groupby(columns[0], dropna=False))
    if not groups:
        return frame.head(0)

    base_sample = sample // len(groups)
    remainder = sample % len(groups)
    parts = []
    shortages = []

    for index, (group_key, group) in enumerate(groups):
        group_sample = base_sample + (1 if index < remainder else 0)
        if len(group) < group_sample:
            shortages.append((group_key, group_sample, len(group)))
            continue
        parts.append(_balancedGroupSample(group, columns[1:], group_sample, random_state))

    if shortages:
        details = ", ".join(
            f"{group_key}: needed {needed}, available {available}"
            for group_key, needed, available in shortages
        )
        raise ValueError(f"Not enough rows to create a balanced subset. {details}")

    return pd.concat(parts, ignore_index=True)


def _stratifiedSample(frame: pd.DataFrame, columns: list[str], sample: int, random_state: int) -> pd.DataFrame:
    if len(frame) <= sample:
        return frame.sample(frac=1, random_state=random_state)

    weights = frame.groupby(columns, dropna=False).size()
    weights = weights / weights.sum()
    parts = []
    remaining = sample
    groups = list(frame.groupby(columns, dropna=False))
    for index, (group_key, group) in enumerate(groups):
        if index == len(groups) - 1:
            group_sample = min(remaining, len(group))
        else:
            group_sample = min(max(1, round(sample * float(weights.loc[group_key]))), len(group), remaining)
        if group_sample > 0:
            parts.append(group.sample(n=group_sample, random_state=random_state))
            remaining -= group_sample

    return pd.concat(parts).sample(frac=1, random_state=random_state).head(sample)


def _toDataFrame(data: Any) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data.copy()

    if isinstance(data, Dataset):
        return data.to_pandas()

    if isinstance(data, dict):
        if data and all(isinstance(value, Dataset) for value in data.values()):
            frames = []
            for split_name, split_data in data.items():
                frame = split_data.to_pandas()
                frame.insert(0, "__split", split_name)
                frames.append(frame)
            return pd.concat(frames, ignore_index=True)

        return pd.DataFrame(data)

    if isinstance(data, list):
        return pd.DataFrame(data)

    if hasattr(data, "to_pandas"):
        return data.to_pandas()

    raise TypeError(f"Unsupported data type for EDA: {type(data).__name__}")


def _printDataFrame(frame: pd.DataFrame, title: str) -> None:
    table = Table(title=title, show_lines=False)
    table.add_column("index", style="dim")
    for column in frame.columns:
        table.add_column(str(column), overflow="fold", max_width=40)

    for index, row in frame.iterrows():
        table.add_row(str(index), *[_formatCell(row[column]) for column in frame.columns])

    _CONSOLE.print(table)


def _printSchema(schema: dict[str, Any]) -> None:
    _CONSOLE.print(Panel(f"shape: {schema['shape']}", title="Dataset Schema"))

    table = Table(title="Columns")
    table.add_column("column")
    table.add_column("dtype")
    table.add_column("type")

    column_types = schema["columnTypes"]
    type_by_column = {
        column: type_name
        for type_name, columns in column_types.items()
        for column in columns
    }

    for column in schema["columns"]:
        table.add_row(column, schema["dtypes"][column], type_by_column.get(column, "unknown"))

    _CONSOLE.print(table)


def _printWarning(message: str) -> None:
    _CONSOLE.print(Panel(message, title="Not plottable"))


def _formatCell(value: Any) -> str:
    text = str(value)
    return text if len(text) <= 120 else text[:117] + "..."


def _describeSummary(row: pd.Series) -> str:
    kind = row.get("kind")
    if kind == "numeric":
        return f"mean={row.get('mean'):.3g}, min={row.get('min'):.3g}, max={row.get('max'):.3g}"
    if kind == "list":
        return f"avg list={row.get('avgListLength'):.3g}, avg text={row.get('avgTextLength'):.3g}"
    if kind == "text":
        return f"avg text={row.get('avgTextLength'):.3g}, top count={row.get('topValueCount')}"
    if kind == "boolean":
        return f"true={row.get('trueCount')}, false={row.get('falseCount')}"
    if kind == "datetime":
        return f"min={row.get('min')}, max={row.get('max')}"

    return ""


def _detectColumnTypes(data: Any) -> dict[str, list[str]]:
    frame = _toDataFrame(data)
    numeric = frame.select_dtypes(include="number").columns.tolist()
    boolean = frame.select_dtypes(include="bool").columns.tolist()
    datetime = frame.select_dtypes(include="datetime").columns.tolist()

    categorical: list[str] = []
    text: list[str] = []
    for column in frame.select_dtypes(include="object").columns:
        try:
            unique_count = frame[column].nunique(dropna=True)
        except TypeError:
            unique_count = frame[column].astype("string").nunique(dropna=True)
        if unique_count <= min(50, max(1, len(frame) // 2)):
            categorical.append(column)
        else:
            text.append(column)

    return {
        "numeric": numeric,
        "boolean": boolean,
        "datetime": datetime,
        "categorical": categorical,
        "text": text,
    }


def _selectPlottableColumns(data: Any) -> list[str]:
    column_types = _detectColumnTypes(data)
    return column_types["numeric"] + column_types["boolean"] + column_types["categorical"]


def _selectFeatureMatrix(data: Any, columns: list[str] | None = None) -> pd.DataFrame:
    frame = _toDataFrame(data)
    if columns is not None:
        return _columnsToFeatureMatrix(frame, columns)

    if "features" in frame.columns and _isListLikeColumn(frame["features"].dropna()):
        features_frame = pd.DataFrame(frame["features"].tolist(), index=frame.index)
        if not features_frame.empty and all(pd.api.types.is_numeric_dtype(features_frame[column]) for column in features_frame.columns):
            return features_frame

    numeric_frame = frame.select_dtypes(include="number")
    if not numeric_frame.empty:
        return numeric_frame

    text_columns = [
        column
        for column in frame.columns
        if pd.api.types.is_string_dtype(frame[column]) or _isListLikeColumn(frame[column].dropna())
    ]
    if text_columns:
        return _textColumnsToFeatureMatrix(frame, text_columns)

    categorical_frame = pd.get_dummies(frame)
    if categorical_frame.empty:
        raise ValueError("No usable columns available for plotting")

    return categorical_frame


def _columnsToFeatureMatrix(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    selected = frame[columns]
    if all(pd.api.types.is_numeric_dtype(selected[column]) for column in selected.columns):
        return selected

    text_columns = [
        column
        for column in selected.columns
        if pd.api.types.is_string_dtype(selected[column]) or _isListLikeColumn(selected[column].dropna())
    ]
    if text_columns:
        return _textColumnsToFeatureMatrix(selected, text_columns)

    return pd.get_dummies(selected)


def _textColumnsToFeatureMatrix(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    text = frame[columns].apply(lambda row: " ".join(_stringifyFeatureValue(value) for value in row), axis=1)
    matrix = TfidfVectorizer(max_features=1000, stop_words="english").fit_transform(text)
    return pd.DataFrame.sparse.from_spmatrix(matrix, index=frame.index)


def _stringifyFeatureValue(value: Any) -> str:
    if pd.api.types.is_list_like(value) and not isinstance(value, (str, bytes, dict)):
        return " ".join(str(item) for item in value)

    if pd.isna(value):
        return ""

    return str(value)


def _projectFeatureMatrix(
    data: Any,
    dimensions: int,
    method: str,
    columns: list[str] | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    if dimensions not in {2, 3}:
        raise ValueError("dimensions must be 2 or 3")

    matrix = _selectFeatureMatrix(data, columns)
    if method == "pca":
        projected = PCA(n_components=dimensions, random_state=random_state).fit_transform(matrix)
    elif method == "tsne":
        projected = TSNE(n_components=dimensions, random_state=random_state, init="random", learning_rate="auto").fit_transform(matrix)
    else:
        raise ValueError("method must be 'pca' or 'tsne'")

    return pd.DataFrame(projected, columns=[f"component{i + 1}" for i in range(dimensions)], index=matrix.index)


def _encodeColors(colors: pd.Series | None) -> Any:
    if colors is None:
        return None

    if pd.api.types.is_numeric_dtype(colors):
        return colors

    return pd.Categorical(colors).codes


def _safeUniqueCount(series: pd.Series) -> int:
    try:
        return int(series.nunique(dropna=True))
    except TypeError:
        return int(series.astype("string").nunique(dropna=True))


def _requireColumns(frame: pd.DataFrame, columns: list[str]) -> None:
    missing_columns = [column for column in columns if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"Columns not found: {missing_columns}")


def _requireNumericColumns(frame: pd.DataFrame, columns: list[str]) -> None:
    _requireColumns(frame, columns)
    non_numeric_columns = [
        column
        for column in columns
        if not pd.api.types.is_numeric_dtype(frame[column])
    ]
    if non_numeric_columns:
        raise ValueError(f"Columns must be numeric: {non_numeric_columns}")


def _toTextSeries(series: pd.Series, column: str) -> pd.Series:
    if _isListLikeColumn(series.dropna()):
        raise ValueError(f"Column '{column}' is list-like. Choose a text/string column instead.")
    return series.fillna("").astype("string")


def _defaultFeatureName(*parts: Any) -> str:
    return "_".join(str(part) for part in parts if part not in (None, ""))


def _safeDivide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.replace(0, pd.NA)
    return numerator / denominator


def _validateThreshold(name: str, value: float) -> None:
    if value < 0:
        raise ValueError(f"{name} must be greater than or equal to 0")


def _isListLikeColumn(series: pd.Series) -> bool:
    if series.empty:
        return False

    sample = series.iloc[: min(20, len(series))]
    return bool(
        sample.map(
            lambda value: pd.api.types.is_list_like(value) and not isinstance(value, (str, bytes, dict))
        ).all()
    )


def _prepareOutputDir(output_dir: str | Path | None) -> Path | None:
    if output_dir is None:
        return None

    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _saveFigure(fig: Any, output_path: str | Path | None) -> Any:
    if output_path is not None:
        fig.savefig(output_path, bbox_inches="tight")

    return fig


def _showFigure(fig: Any, show: bool) -> None:
    if show:
        plt.figure(fig.number)
        plt.show(block=True)


def _getScaler(method: str) -> StandardScaler | MinMaxScaler | RobustScaler:
    if method == "standard":
        return StandardScaler()
    if method == "minmax":
        return MinMaxScaler()
    if method == "robust":
        return RobustScaler()

    raise ValueError("method must be 'standard', 'minmax', or 'robust'")


def _setActiveDatasetState(
    name: str,
    source: str,
    config_name: str | None,
    split: str | None,
    output_path: str | Path | None,
    save_as: str | None,
    data: Any,
) -> None:
    global _ACTIVE_DATASET_NAME, _ACTIVE_DATASET_SOURCE, _ACTIVE_DATASET_CONFIG_NAME
    global _ACTIVE_DATASET_SPLIT, _ACTIVE_DATASET_OUTPUT_PATH, _ACTIVE_DATASET_SAVE_AS, _ACTIVE_DATASET

    _ACTIVE_DATASET_NAME = name
    _ACTIVE_DATASET_SOURCE = source
    _ACTIVE_DATASET_CONFIG_NAME = config_name
    _ACTIVE_DATASET_SPLIT = split
    _ACTIVE_DATASET_OUTPUT_PATH = output_path
    _ACTIVE_DATASET_SAVE_AS = save_as
    _ACTIVE_DATASET = data


def _safeDatasetName(name: str) -> str:
    return name.replace("/", "__").replace(":", "__")


def _datasetStorageName(name: str, config_name: str | None, save_as: str | None) -> str:
    if save_as is not None:
        return _safeDatasetName(save_as)

    if config_name is not None:
        return _safeDatasetName(f"{name}__{config_name}")

    return _safeDatasetName(name)


def _resolveOutputPath(output_path: str | Path | None) -> Path:
    if output_path is None:
        return DATASETS_DIR

    path = Path(output_path).expanduser()
    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def _datasetStoragePath(
    name: str,
    config_name: str | None,
    output_path: str | Path | None,
    save_as: str | None,
) -> Path:
    return _resolveOutputPath(output_path) / _datasetStorageName(name, config_name, save_as)


def _getHuggingfaceToken() -> str | None:
    for env_name in ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        token = os.environ.get(env_name)
        if token:
            return token

    if not ENV_FILE.exists():
        return None

    with ENV_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            key, separator, value = line.strip().partition("=")
            if separator and key in {"HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGINGFACE_HUB_TOKEN"}:
                return value.strip().strip("\"'")

    return None


def _downloadFromHuggingface(name: str, config_name: str | None, split: str | None, **kwargs: Any) -> Any:
    kwargs.setdefault("cache_dir", str(HUGGINGFACE_CACHE_DIR))
    token = _getHuggingfaceToken()
    if token is not None:
        kwargs.setdefault("token", token)

    if split is None:
        return load_dataset(name, config_name, **kwargs)

    return load_dataset(name, config_name, split=split, **kwargs)


def _downloadFromSklearn(name: str) -> Dataset:
    if name not in _SKLEARN_LOADERS:
        available = ", ".join(sorted(_SKLEARN_LOADERS))
        raise ValueError(f"Unknown sklearn dataset '{name}'. Available: {available}")

    raw = _SKLEARN_LOADERS[name]()
    rows: dict[str, Any] = {"features": raw.data.tolist()}

    if hasattr(raw, "target"):
        rows["target"] = raw.target.tolist()

    return Dataset.from_dict(rows)


def _saveToLocal(
    name: str,
    data: Any,
    config_name: str | None = None,
    output_path: str | Path | None = None,
    save_as: str | None = None,
) -> Path:
    dataset_path = _datasetStoragePath(name, config_name, output_path, save_as)
    dataset_path.mkdir(parents=True, exist_ok=True)
    data.save_to_disk(str(dataset_path / "data"))
    return dataset_path
