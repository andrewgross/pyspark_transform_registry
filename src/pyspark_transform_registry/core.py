"""
Simplified PySpark Transform Registry Core Module

This module provides a clean, simple interface for registering and loading
PySpark transform functions using MLflow's model registry.
"""

import importlib.util
import os
from typing import Callable, Optional, Union, Any

import mlflow
import mlflow.pyfunc
import mlflow.models
from pyspark.sql import DataFrame

from .model_wrapper import PySparkTransformModel


def register_function(
    func: Optional[Callable] = None,
    *,
    name: str,
    file_path: Optional[str] = None,
    function_name: Optional[str] = None,
    input_example: Optional[DataFrame] = None,
    description: Optional[str] = None,
    extra_pip_requirements: Optional[list[str]] = None,
    tags: Optional[dict[str, Any]] = None,
) -> str:
    """
    Register a PySpark transform function in MLflow's model registry.

    Supports two modes:
    1. Direct function registration: pass the function directly
    2. File-based registration: load function from Python file

    Args:
        func: The function to register (for direct registration)
        name: Model name for registry (supports 3-part naming: catalog.schema.table)
        file_path: Path to Python file containing the function (for file-based registration)
        function_name: Name of function to extract from file (required for file-based)
        input_example: Sample input DataFrame for signature inference
        description: Model description
        extra_pip_requirements: Additional pip requirements beyond auto-detected ones
        tags: Tags to attach to the registered model

    Returns:
        Model URI of the registered model

    Examples:
        # Direct function registration
        >>> def my_transform(df: DataFrame) -> DataFrame:
        ...     return df.select("*")
        >>> register_function(my_transform, name="my_catalog.my_schema.my_transform")

        # File-based registration
        >>> register_function(
        ...     file_path="transforms/my_transform.py",
        ...     function_name="my_transform",
        ...     name="my_catalog.my_schema.my_transform"
        ... )
    """
    # Validate input arguments
    if func is None and file_path is None:
        raise ValueError("Either 'func' or 'file_path' must be provided")

    if func is not None and file_path is not None:
        raise ValueError("Cannot specify both 'func' and 'file_path'")

    if file_path is not None and function_name is None:
        raise ValueError("'function_name' is required when using 'file_path'")

    # Load function from file if needed
    if file_path is not None:
        func = _load_function_from_file(file_path, function_name)

    # Create model wrapper
    model = PySparkTransformModel(func)

    # Prepare MLflow logging parameters
    log_params = {
        "python_model": model,
        "registered_model_name": name,
        "infer_code_paths": True,  # Auto-detect Python modules
        "extra_pip_requirements": extra_pip_requirements,
        "tags": tags or {},
    }

    # Add input example and infer signature if provided
    if input_example is not None:
        try:
            # Convert Spark DataFrame to pandas for MLflow
            pandas_input_example = input_example.toPandas()
            log_params["input_example"] = pandas_input_example

            # Infer signature using original DataFrames
            output_example = model.predict(input_example)
            pandas_output_example = output_example.toPandas()
            log_params["signature"] = mlflow.models.infer_signature(
                pandas_input_example,
                pandas_output_example,
            )
        except Exception as e:
            # If pandas conversion fails, skip input example but log without it
            print(f"Warning: Could not convert input example to pandas, skipping: {e}")
            pass

    # Add description as metadata
    if description:
        log_params["tags"]["description"] = description

    # Add function metadata
    log_params["tags"]["function_name"] = func.__name__
    if func.__doc__:
        log_params["tags"]["docstring"] = func.__doc__

    # Log the model
    with mlflow.start_run():
        # Use a simple artifact name but register with the full name
        artifact_name = func.__name__ if func else function_name
        mlflow.pyfunc.log_model(artifact_path=artifact_name, **log_params)

    # Return the model URI string
    return f"models:/{name}/1"


def load_function(name: str, version: Optional[Union[int, str]] = None) -> Callable:
    """
    Load a previously registered PySpark transform function.

    Args:
        name: Model name in registry (supports 3-part naming: catalog.schema.table)
        version: Model version to load (defaults to latest if not specified)

    Returns:
        The loaded transform function

    Examples:
        # Load latest version
        >>> transform = load_function("my_catalog.my_schema.my_transform")

        # Load specific version
        >>> transform = load_function("my_catalog.my_schema.my_transform", version=2)
    """
    # Build model URI
    if version is None:
        model_uri = f"models:/{name}/latest"
    else:
        model_uri = f"models:/{name}/{version}"

    # Load the model
    loaded_model = mlflow.pyfunc.load_model(model_uri)

    # Return the predict function which wraps our transform
    return loaded_model.predict


def list_registered_functions(name_prefix: Optional[str] = None) -> list[dict]:
    """
    List registered transform functions.

    Args:
        name_prefix: Optional prefix to filter model names

    Returns:
        List of registered models with their metadata
    """
    client = mlflow.tracking.MlflowClient()

    # Get all registered models
    models = client.search_registered_models()

    # Filter by prefix if provided
    if name_prefix:
        models = [m for m in models if m.name.startswith(name_prefix)]

    # Return simplified model info
    return [
        {
            "name": model.name,
            "latest_version": model.latest_versions[0].version
            if model.latest_versions
            else None,
            "description": model.description,
            "tags": model.tags,
        }
        for model in models
    ]


def _load_function_from_file(file_path: str, function_name: str) -> Callable:
    """
    Load a function from a Python file.

    Args:
        file_path: Path to the Python file
        function_name: Name of the function to extract

    Returns:
        The loaded function
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # Load the module
    spec = importlib.util.spec_from_file_location("transform_module", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Get the function
    if not hasattr(module, function_name):
        raise AttributeError(f"Function '{function_name}' not found in {file_path}")

    func = getattr(module, function_name)

    if not callable(func):
        raise TypeError(f"'{function_name}' is not a function")

    return func
