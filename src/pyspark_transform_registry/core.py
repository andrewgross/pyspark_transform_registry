import datetime
import importlib.util
import inspect
import json
import os
import tempfile
from typing import Callable

import mlflow
from pyspark.sql import SparkSession

from .metadata import _get_function_metadata, _wrap_function_source

spark = SparkSession.getActiveSession()


def log_transform_function(
    func: Callable,
    name: str,
    artifact_path: str = "transform_code",
    as_text: bool = True,
):
    """
    Logs a PySpark transform function's source code to MLflow, with metadata and docstring header.
    """
    source = inspect.getsource(func)
    param_info, return_type, doc = _get_function_metadata(func)
    wrapped_source = _wrap_function_source(name, source, doc, param_info, return_type)
    filename = f"{name}.py"

    # Log the source code
    if as_text:
        # Create a temporary file to ensure the directory exists
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, filename)
            with open(path, "w") as f:
                f.write(wrapped_source)
            mlflow.log_artifact(path, artifact_path)
    else:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, filename)
            with open(path, "w") as f:
                f.write(wrapped_source)
            mlflow.log_artifact(path, artifact_path)

    # Log metadata using different MLflow features
    # Tags for categorization
    mlflow.set_tag("transform_type", "pyspark")

    # Parameters for structured data
    mlflow.log_param("transform_name", name)
    mlflow.log_param("return_type", return_type or "unspecified")
    mlflow.log_param("param_info", json.dumps(param_info))

    # Log docstring as a parameter (since it might be long)
    mlflow.log_param("docstring", doc)

    # Log timestamp as a metric for versioning
    mlflow.log_metric(
        "timestamp",
        datetime.datetime.now(datetime.timezone.utc).timestamp(),
    )


def load_transform_function(
    run_id: str,
    name: str,
    artifact_path: str = "transform_code",
) -> Callable:
    """
    Loads a logged transform function from an MLflow run using importlib.
    """
    filename = f"{artifact_path}/{name}.py"
    path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=filename)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Could not create module spec for {name}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    func = getattr(mod, name, None)
    if not callable(func):
        raise ValueError(f"No callable named '{name}' found in artifact module.")
    return func


def find_transform_versions(name: str = None, return_type: str = None):
    """
    Find transform versions using MLflow search capabilities.

    Args:
        name: Optional filter by transform name
        return_type: Optional filter by return type

    Returns:
        List of MLflow runs containing the transforms
    """
    # Use parameters for structured data search
    filter_string = "tags.transform_type = 'pyspark'"
    if name is not None:
        filter_string += f" AND params.transform_name = '{name}'"
    if return_type is not None:
        filter_string += f" AND params.return_type = '{return_type}'"

    return mlflow.search_runs(filter_string=filter_string)