from __future__ import annotations

import os
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASETS_DIR = PROJECT_ROOT / "Datasets"
HUGGINGFACE_CACHE_DIR = DATASETS_DIR / "_huggingface_cache"
ENV_FILE = PROJECT_ROOT / ".env"

os.environ["HF_HOME"] = str(HUGGINGFACE_CACHE_DIR / "home")
os.environ["HF_HUB_CACHE"] = str(HUGGINGFACE_CACHE_DIR / "hub")
os.environ["HF_DATASETS_CACHE"] = str(HUGGINGFACE_CACHE_DIR / "datasets")

from datasets import Dataset, load_dataset, load_from_disk
import matplotlib.pyplot as plt
import pandas as pd
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

    if dataset_path.suffix == ".parquet":
        return pd.read_parquet(dataset_path)

    if dataset_path.suffix == ".csv":
        return pd.read_csv(dataset_path)

    if dataset_path.suffix == ".json":
        return pd.read_json(dataset_path)

    return None


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
