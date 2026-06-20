from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from Core import withSpinner

from .utils import (
    _SKLEARN_LOADERS,
    _createDatasetSubset,
    _datasetStoragePath,
    _resolveOutputPath,
    _getActiveDatasetConfig,
    _getMatchingActiveDataset,
    _downloadFromHuggingface,
    _downloadFromSklearn,
    _loadLocalDataset,
    _requireActiveDatasetName,
    _saveToLocal,
    _setActiveDatasetState,
)

from datasets import get_dataset_infos, load_from_disk


@withSpinner()
def createDatasetSubset(
    name: str,
    saveAs: str,
    sample: int,
    outputPath: str | Path | None = None,
    split: str | None = None,
    columns: list[str] | None = None,
    stratifyBy: str | list[str] | None = None,
    balance: bool = False,
    randomState: int = 42,
) -> Any:
    """Create a sampled local dataset subset under Datasets/."""
    subset, dataset_path = _createDatasetSubset(
        name=name,
        save_as=saveAs,
        sample=sample,
        output_path=outputPath,
        split=split,
        columns=columns,
        stratify_by=stratifyBy,
        balance=balance,
        random_state=randomState,
    )
    metadata = {
        "name": name,
        "source": "local_subset",
        "outputPath": str(_resolveOutputPath(outputPath)),
        "saveAs": saveAs,
        "sample": sample,
        "split": split,
        "columns": columns,
        "stratifyBy": stratifyBy,
        "balance": balance,
        "randomState": randomState,
        "path": str(dataset_path),
        "features": {column: str(dtype) for column, dtype in subset.dtypes.items()},
        "numRows": int(len(subset)),
    }
    with (dataset_path / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)

    return subset


@withSpinner()
def setActiveDataset(
    name: str,
    outputPath: str | Path | None = None,
) -> Any:
    """Set the local dataset used by Data APIs."""
    data = _loadLocalDataset(name, outputPath)
    if data is None:
        raise ValueError(f"Dataset '{name}' was not found under {_resolveOutputPath(outputPath)}")

    _setActiveDatasetState(name, "local", None, None, outputPath, None, data)
    return data


@withSpinner()
def downloadDataset(
    name: str | None = None,
    source: str = "huggingface",
    configName: str | None = None,
    split: str | None = None,
    outputPath: str | Path | None = None,
    saveAs: str | None = None,
    cache: bool = True,
    force: bool = False,
    **kwargs: Any,
) -> Any:
    """Download or load a dataset and optionally cache it under Datasets/."""
    using_active_dataset = name is None
    name = _requireActiveDatasetName(name)

    if using_active_dataset:
        source, configName, split, outputPath, saveAs = _getActiveDatasetConfig(source)

    active_dataset = _getMatchingActiveDataset(name, source, configName, split, outputPath, saveAs, force)
    if active_dataset is not None:
        return active_dataset

    dataset_path = _datasetStoragePath(name, configName, outputPath, saveAs)
    local_path = dataset_path / "data"
    if cache and not force and local_path.exists():
        try:
            return load_from_disk(str(local_path))
        except FileNotFoundError:
            pass

    if source == "huggingface":
        data = _downloadFromHuggingface(name, configName, split, **kwargs)
    elif source == "sklearn":
        data = _downloadFromSklearn(name)
    else:
        raise ValueError("source must be 'huggingface' or 'sklearn'")

    if cache:
        dataset_path = _saveToLocal(name, data, configName, outputPath, saveAs)
        features: dict[str, Any] = {}
        num_rows: int | dict[str, int] | None = None

        if hasattr(data, "features"):
            features = {key: str(value) for key, value in data.features.items()}
            num_rows = int(data.num_rows)
        elif isinstance(data, dict):
            features = {
                split_name: {key: str(value) for key, value in dataset.features.items()}
                for split_name, dataset in data.items()
                if hasattr(dataset, "features")
            }
            num_rows = {
                split_name: int(dataset.num_rows)
                for split_name, dataset in data.items()
                if hasattr(dataset, "num_rows")
            }

        metadata = {
            "name": name,
            "source": source,
            "configName": configName,
            "split": split,
            "outputPath": str(_resolveOutputPath(outputPath)),
            "saveAs": saveAs,
            "task": "classification" if "label" in str(features).lower() or "target" in str(features).lower() else None,
            "path": str(dataset_path),
            "features": features,
            "numRows": num_rows,
        }
        with (dataset_path / "metadata.json").open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, sort_keys=True)

    return data


@withSpinner()
def listAvailableDatasets(
    outputPath: str | Path | None = None,
) -> list[str]:
    """List top-level dataset files and folders saved under Datasets/."""
    datasets_dir = _resolveOutputPath(outputPath)
    if not datasets_dir.exists():
        print([])
        return []

    datasets = sorted(
        path.name
        for path in datasets_dir.iterdir()
        if not path.name.startswith((".", "_"))
    )
    print(datasets)
    return datasets


@withSpinner()
def getDatasetInfo(
    name: str | None = None,
    configName: str | None = None,
    outputPath: str | Path | None = None,
    saveAs: str | None = None,
) -> dict[str, Any]:
    """Return local, sklearn, or pre-download Hugging Face dataset metadata."""
    using_active_dataset = name is None
    name = _requireActiveDatasetName(name)

    if using_active_dataset and outputPath is None and saveAs is None:
        _, configName, _, outputPath, saveAs = _getActiveDatasetConfig("huggingface")

    metadata_path = _datasetStoragePath(name, configName, outputPath, saveAs) / "metadata.json"
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    if name in _SKLEARN_LOADERS:
        raw = _SKLEARN_LOADERS[name]()
        return {
            "name": name,
            "source": "sklearn",
            "task": "classification" if hasattr(raw, "target_names") else "regression",
            "features": list(getattr(raw, "feature_names", [])),
            "numRows": int(raw.data.shape[0]),
            "numFeatures": int(raw.data.shape[1]),
        }

    infos = get_dataset_infos(name)
    configs: dict[str, Any] = {}

    for config_name, info in infos.items():
        splits = info.splits or {}
        configs[config_name] = {
            "description": info.description,
            "features": {key: str(value) for key, value in info.features.items()}
            if info.features
            else {},
            "task": "classification" if "label" in str(info.features).lower() else None,
            "splits": {
                split_name: {
                    "numExamples": split_info.num_examples,
                    "numBytes": getattr(split_info, "num_bytes", None),
                }
                for split_name, split_info in splits.items()
            },
            "datasetSize": getattr(info, "dataset_size", None),
            "downloadSize": getattr(info, "download_size", None),
        }

    return {
        "name": name,
        "source": "huggingface",
        "configs": configs,
    }
