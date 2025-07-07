import datetime
import importlib.util
import inspect
import json
import os
import tempfile
from typing import Callable, List, Optional

import mlflow
from pyspark.sql import SparkSession

from .metadata import _get_function_metadata, _wrap_function_source
from .versioning import validate_semver, normalize_version, matches_version_constraint

spark = SparkSession.getActiveSession()


def log_transform_function(
    func: Callable,
    name: str,
    version: Optional[str] = None,
    artifact_path: str = "transform_code",
    as_text: bool = True,
    allow_version_overwrite: bool = False,
):
    """
    Logs a PySpark transform function's source code to MLflow, with metadata and docstring header.
    
    Args:
        func: The function to log
        name: Name for the logged function
        version: Version string in SemVer format (e.g., "1.2.3"). If None, defaults to "0.1.0"
        artifact_path: Path where to store the artifact
        as_text: If True, logs source code directly as text using mlflow.log_text().
                 If False, logs source code as a file artifact using mlflow.log_artifact().
        allow_version_overwrite: If True, allows overwriting existing name+version combinations.
                                If False, raises error if name+version already exists.
    
    Raises:
        ValueError: If version is not valid SemVer format or if name+version already exists
                   and allow_version_overwrite is False
    """
    # Handle version parameter
    if version is None:
        version = "0.1.0"
    
    # Validate version format
    if not validate_semver(version):
        raise ValueError(f"Invalid SemVer format: {version}")
    
    # Normalize version for consistency
    version = normalize_version(version)
    
    # Check for existing name+version combination
    if not allow_version_overwrite:
        existing = find_transform_versions(name=name, version=version)
        if len(existing) > 0:
            raise ValueError(f"Transform '{name}' version '{version}' already exists. "
                           f"Use allow_version_overwrite=True to overwrite.")
    
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

    # Log metadata using different MLflow features
    # Tags for categorization
    mlflow.set_tag("transform_type", "pyspark")

    # Parameters for structured data
    mlflow.log_param("transform_name", name)
    mlflow.log_param("version", version)
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


def find_transform_versions(
    name: Optional[str] = None, 
    return_type: Optional[str] = None,
    version: Optional[str] = None,
    version_constraint: Optional[str] = None,
):
    """
    Find transform versions using MLflow search capabilities.

    Args:
        name: Optional filter by transform name
        return_type: Optional filter by return type
        version: Optional filter by exact version
        version_constraint: Optional version constraint (e.g., ">=1.0.0", "~=1.2.0")

    Returns:
        List of MLflow runs containing the transforms
    """
    # Use parameters for structured data search
    filter_string = "tags.transform_type = 'pyspark'"
    if name is not None:
        filter_string += f" AND params.transform_name = '{name}'"
    if return_type is not None:
        filter_string += f" AND params.return_type = '{return_type}'"
    if version is not None:
        filter_string += f" AND params.version = '{version}'"

    runs = mlflow.search_runs(filter_string=filter_string)
    
    # Apply version constraint filtering if specified  
    # Note: Version constraint filtering is applied post-search since MLflow
    # doesn't support complex version operators in its query language
    if version_constraint is not None and len(runs) > 0:
        # MLflow returns a DataFrame, so we filter it appropriately
        if hasattr(runs, 'iterrows'):  # DataFrame case
            filtered_indices = []
            for idx, row in runs.iterrows():
                run_version = row.get('params.version')
                if run_version is not None:
                    try:
                        if matches_version_constraint(run_version, version_constraint):
                            filtered_indices.append(idx)
                    except ValueError:
                        continue
            return runs.loc[filtered_indices] if filtered_indices else runs.iloc[0:0]
        else:  # List case - just return as is for now
            return runs
    
    return runs


def get_transform_versions(name: str) -> List[str]:
    """
    Get all versions of a specific transform without downloading the code.
    
    Args:
        name: Name of the transform to get versions for
        
    Returns:
        List of version strings, sorted from oldest to newest
        
    Raises:
        ValueError: If no transform with the given name exists
    """
    runs = find_transform_versions(name=name)
    
    if len(runs) == 0:
        raise ValueError(f"No transform found with name: {name}")
    
    versions = []
    # Handle both DataFrame and list cases
    if hasattr(runs, 'iterrows'):  # DataFrame case
        for _, row in runs.iterrows():
            version = row.get('params.version')
            if version:
                versions.append(version)
    else:  # List case
        for run in runs:
            version = getattr(run, 'params', {}).get('version')
            if version:
                versions.append(version)
    
    # Sort versions using semantic version comparison
    from .versioning import get_latest_version
    try:
        # Use packaging to sort versions properly
        from packaging.version import Version
        versions = [str(v) for v in sorted([Version(v) for v in versions])]
    except Exception:
        # Fallback to string sorting if version parsing fails
        versions = sorted(versions)
    
    return versions


def get_latest_transform_version(name: str) -> Optional[str]:
    """
    Get the latest version of a specific transform.
    
    Args:
        name: Name of the transform
        
    Returns:
        Latest version string, or None if no transform with the name exists
    """
    try:
        versions = get_transform_versions(name)
        if versions:
            from .versioning import get_latest_version
            return get_latest_version(versions)
        return None
    except ValueError:
        return None


def load_transform_function_by_version(
    name: str,
    version: str = "latest",
    artifact_path: str = "transform_code",
) -> Callable:
    """
    Load a transform function by name and version.
    
    Args:
        name: Name of the transform function
        version: Version to load ("latest" for most recent, or specific version like "1.2.3")
        artifact_path: Path where the artifact is stored
        
    Returns:
        The loaded transform function
        
    Raises:
        ValueError: If transform with specified name/version not found
    """
    if version == "latest":
        actual_version = get_latest_transform_version(name)
        if actual_version is None:
            raise ValueError(f"No versions found for transform: {name}")
        version = actual_version
    
    # Find the specific run with this name and version
    runs = find_transform_versions(name=name, version=version)
    
    if len(runs) == 0:
        raise ValueError(f"No transform found with name '{name}' and version '{version}'")
    
    # Get the run_id from the first matching run
    if hasattr(runs, 'iloc'):  # DataFrame case
        run_id = runs.iloc[0]['run_id']
    else:  # List case
        run_id = runs[0].info.run_id
    
    return load_transform_function(run_id, name, artifact_path)