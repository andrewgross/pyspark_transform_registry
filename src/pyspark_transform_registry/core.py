import datetime
import importlib.util
import inspect
import json
import os
import tempfile
from typing import Callable, Optional

import mlflow
from pyspark.sql import SparkSession

from .metadata import _get_function_metadata, _wrap_function_source
from .versioning import generate_next_version, parse_semantic_version

spark = SparkSession.getActiveSession()


def log_transform_function(
    func: Callable,
    name: str,
    artifact_path: str = "transform_code",
    as_text: bool = True,
    version: Optional[str] = None,
    version_bump: Optional[str] = None,
):
    """
    Logs a PySpark transform function's source code to MLflow, with metadata and versioning.

    Args:
        func: The function to log
        name: Name for the logged function
        artifact_path: Path where to store the artifact
        as_text: If True, logs source code directly as text using mlflow.log_text().
                 If False, logs source code as a file artifact using mlflow.log_artifact().
        version: Optional explicit semantic version (e.g., "1.2.0")
        version_bump: Optional explicit version bump type ("major", "minor", "patch")
    """
    source = inspect.getsource(func)
    param_info, return_type, doc = _get_function_metadata(func)
    wrapped_source = _wrap_function_source(name, source, doc, param_info, return_type)
    filename = f"{name}.py"

    # Log the source code
    if as_text:
        # Log source code directly as text artifact
        mlflow.log_text(wrapped_source, f"{artifact_path}/{filename}")
    else:
        # Log source code as file artifact
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, filename)
            with open(path, "w") as f:
                f.write(wrapped_source)
            mlflow.log_artifact(path, artifact_path)

    # Handle versioning
    if version is not None:
        # Validate and use explicit version
        semantic_version = parse_semantic_version(version)
    else:
        # Auto-generate version based on interface changes
        semantic_version = generate_next_version(name, func, version_bump)

    # Log metadata using different MLflow features
    # Tags for categorization and versioning
    mlflow.set_tag("transform_type", "pyspark")
    mlflow.set_tag("transform_name", name)
    mlflow.set_tag("semantic_version", str(semantic_version))

    # Parameters for structured data
    mlflow.log_param("transform_name", name)
    mlflow.log_param("return_type", return_type or "unspecified")
    mlflow.log_param("param_info", json.dumps(param_info))
    mlflow.log_param("major_version", semantic_version.major)
    mlflow.log_param("minor_version", semantic_version.minor)
    mlflow.log_param("patch_version", semantic_version.patch)

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


def find_transform_versions(
    name: str = None,
    return_type: str = None,
    version_constraint: str = None,
    latest_only: bool = False,
):
    """
    Find transform versions using MLflow search capabilities.

    Args:
        name: Optional filter by transform name
        return_type: Optional filter by return type
        version_constraint: Optional version constraint (e.g., ">=1.0.0,<2.0.0")
        latest_only: If True, return only the latest version

    Returns:
        List of MLflow runs containing the transforms
    """
    # Use parameters for structured data search
    filter_string = "tags.transform_type = 'pyspark'"
    if name is not None:
        filter_string += f" AND params.transform_name = '{name}'"
    if return_type is not None:
        filter_string += f" AND params.return_type = '{return_type}'"

    # Get all matching runs
    runs = mlflow.search_runs(filter_string=filter_string, order_by=["start_time DESC"])

    # Apply version constraint filtering if specified
    if version_constraint is not None:
        from .versioning import satisfies_version_constraint

        filtered_indices = []
        for idx, run in runs.iterrows():
            semantic_version_col = "tags.semantic_version"
            if semantic_version_col in run and run[semantic_version_col] is not None:
                try:
                    version = parse_semantic_version(run[semantic_version_col])
                    if satisfies_version_constraint(version, version_constraint):
                        filtered_indices.append(idx)
                except ValueError:
                    # Skip invalid version formats
                    continue
        runs = runs.loc[filtered_indices] if filtered_indices else runs.iloc[0:0]

    # Return only latest version if requested
    if latest_only and len(runs) > 0:
        # Find the run with the highest semantic version
        latest_idx = None
        latest_version = None

        for idx, run in runs.iterrows():
            semantic_version_col = "tags.semantic_version"
            if semantic_version_col in run and run[semantic_version_col] is not None:
                try:
                    version = parse_semantic_version(run[semantic_version_col])
                    if latest_version is None or version > latest_version:
                        latest_version = version
                        latest_idx = idx
                except ValueError:
                    continue

        return runs.loc[[latest_idx]] if latest_idx is not None else runs.iloc[0:0]

    return runs
