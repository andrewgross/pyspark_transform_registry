"""
Simplified PySpark Transform Registry Core Module

This module provides a clean, simple interface for registering and loading
PySpark transform functions using MLflow's model registry.
"""

import importlib.util
import logging
import os
from collections.abc import Callable
from typing import Any

import mlflow
import mlflow.pyfunc
from mlflow.models.model import ModelInfo
from pyspark.sql import DataFrame

from .model_wrapper import PySparkTransformModel
from .runtime_validation import RuntimeValidator
from .schema_constraints import PartialSchemaConstraint
from .static_analysis import analyze_function

# Configure logger for this module
logger = logging.getLogger(__name__)


def register_function(
    func: Callable | None = None,
    *,
    name: str,
    file_path: str | None = None,
    function_name: str | None = None,
    description: str | None = None,
    extra_pip_requirements: list[str] | None = None,
    tags: dict[str, Any] | None = None,
    infer_schema: bool = True,
    schema_constraint: PartialSchemaConstraint | None = None,
) -> ModelInfo:
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
        description: Model description
        extra_pip_requirements: Additional pip requirements beyond auto-detected ones
        tags: Tags to attach to the registered model
        infer_schema: Whether to automatically infer schema constraints using static analysis
        schema_constraint: Pre-computed schema constraint (overrides infer_schema if provided)

    Returns:
        Model URI of the registered model

    Examples:
        # Direct function registration
        >>> def my_transform(df: DataFrame) -> DataFrame:
        ...     return df.select("*")
        >>> register_function(my_transform, name="my_catalog.my_schema.my_transform")

        # Multi-parameter function registration
        >>> def filter_transform(df: DataFrame, min_value: int = 0) -> DataFrame:
        ...     return df.filter(df.value >= min_value)
        >>> register_function(
        ...     filter_transform,
        ...     name="my_catalog.my_schema.filter_transform",
        ... )

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

    # Perform schema inference if requested
    inferred_constraint = None
    if schema_constraint is not None:
        # Use provided constraint
        inferred_constraint = schema_constraint
    elif infer_schema:
        # Automatically infer schema constraint using static analysis
        try:
            inferred_constraint = analyze_function(func)
        except Exception as e:
            logger.warning(
                "Schema inference failed, proceeding without constraint: %s",
                e,
            )
            inferred_constraint = None

    # Create model wrapper
    model = PySparkTransformModel(func, schema_constraint=inferred_constraint)

    # Prepare MLflow logging parameters
    log_params = {
        "python_model": model,
        "registered_model_name": name,
        "infer_code_paths": True,  # Auto-detect Python modules
        "extra_pip_requirements": extra_pip_requirements,
        "tags": tags or {},
    }

    # Add description as metadata
    if description:
        log_params["tags"]["description"] = description

    # Add function metadata
    _func_name = function_name if function_name else func.__name__
    log_params["tags"]["function_name"] = _func_name
    if func.__doc__:
        log_params["tags"]["docstring"] = func.__doc__

    # Add schema constraint metadata
    if inferred_constraint is not None:
        log_params["tags"]["schema_constraint"] = inferred_constraint.to_json()
        log_params["tags"]["schema_analysis_method"] = (
            inferred_constraint.analysis_method
        )
        log_params["tags"]["schema_required_columns"] = len(
            inferred_constraint.required_columns,
        )
        log_params["tags"]["schema_added_columns"] = len(
            inferred_constraint.added_columns,
        )
        log_params["tags"]["schema_preserves_others"] = str(
            inferred_constraint.preserves_other_columns,
        )

        if inferred_constraint.warnings:
            log_params["tags"]["schema_warnings"] = "; ".join(
                inferred_constraint.warnings,
            )

    # Log the model - handle nested runs if already in a run
    try:
        # Check if there's already an active run
        active_run = mlflow.active_run()
        if active_run is not None:
            # Use nested run
            with mlflow.start_run(nested=True):
                # Set tags explicitly in the run
                for tag_key, tag_value in log_params["tags"].items():
                    mlflow.set_tag(tag_key, tag_value)

                # Use the registered model name for MLflow 3.0+
                logged_model = mlflow.pyfunc.log_model(name=_func_name, **log_params)
        else:
            # Start a new run
            with mlflow.start_run():
                # Set tags explicitly in the run
                for tag_key, tag_value in log_params["tags"].items():
                    mlflow.set_tag(tag_key, tag_value)

                # Use the registered model name for MLflow 3.0+
                logged_model = mlflow.pyfunc.log_model(name=_func_name, **log_params)
    except Exception:
        # Fallback: try with nested=True
        with mlflow.start_run(nested=True):
            # Set tags explicitly in the run
            for tag_key, tag_value in log_params["tags"].items():
                mlflow.set_tag(tag_key, tag_value)

            # Use a simple artifact name but register with the full name
            logged_model = mlflow.pyfunc.log_model(name=_func_name, **log_params)

    return logged_model


def load_function(
    name: str,
    version: int | str,
    validate_input: bool = True,
    strict_validation: bool = False,
) -> Callable:
    """
    Load a previously registered PySpark transform function with optional validation.

    Args:
        name: Model name in registry (supports 3-part naming: catalog.schema.table)
        version: Model version to load (required)
        validate_input: Whether to validate input DataFrames against stored schema constraints
        strict_validation: If True, treat validation warnings as errors

    Returns:
        A callable that supports both single and multi-parameter usage:
        - Single param: transform(df)
        - Multi param: transform(df, params={'param1': value1, 'param2': value2})

        The returned function also has additional methods:
        - transform.get_source(): Returns the source code of the original function
        - transform.get_original_function(): Returns the unwrapped original function

        If validation is enabled, input DataFrames will be validated against the
        stored schema constraints before transformation.

    Examples:
        # Load specific version with validation
        >>> transform = load_function("my_catalog.my_schema.my_transform", version=1)

        # Use with single parameter (validates input automatically)
        >>> result = transform(df)

        # Use with multiple parameters (validates input automatically)
        >>> result = transform(df, params={'min_value': 10, 'threshold': 0.5})

        # Inspect the original source code
        >>> print(transform.get_source())

        # Get the original function for advanced inspection
        >>> original_func = transform.get_original_function()
        >>> import inspect
        >>> print(inspect.signature(original_func))

        # Load specific version without validation
        >>> transform = load_function("my_catalog.my_schema.my_transform", version=2, validate_input=False)

        # Load with strict validation (warnings become errors)
        >>> transform = load_function("my_catalog.my_schema.my_transform", version=1, strict_validation=True)
    """
    # Build model URI with explicit version
    model_uri = f"models:/{name}/{version}"

    # Load the model
    loaded_model = mlflow.pyfunc.load_model(model_uri)

    # Get the original transform function directly from the model wrapper
    # This bypasses MLflow's predict() method which doesn't handle params properly
    # Access the underlying python model via _model_impl.python_model
    original_func = loaded_model._model_impl.python_model.get_transform_function()

    # Extract schema constraint from model metadata if validation is enabled
    schema_constraint = None
    if validate_input:
        schema_constraint = _load_schema_constraint(name, version)

    # Create a wrapper function that handles both single and multi-parameter calls
    def transform_wrapper(df: DataFrame, params: dict | None = None):
        """
        Wrapper function that supports both single and multi-parameter usage with optional validation.

        Args:
            df: Input PySpark DataFrame
            params: Optional dictionary of additional parameters for multi-input functions

        Returns:
            Transformed DataFrame

        Raises:
            ValueError: If input validation fails and strict_validation is True
        """
        # Perform input validation if enabled and constraint is available
        if validate_input and schema_constraint is not None:
            validator = RuntimeValidator(strict_mode=strict_validation)
            validation_result = validator.validate_dataframe(df, schema_constraint)

            if not validation_result.is_valid:
                error_messages = validation_result.get_error_messages()
                raise ValueError(
                    f"Input validation failed: {'; '.join(error_messages)}",
                )

            # Log warnings if any (but don't fail the execution)
            warnings = validation_result.get_warning_messages()
            if warnings:
                for warning in warnings:
                    print(f"Validation warning: {warning}")

        # Execute the transform function
        if params is None:
            # Single parameter function call
            return original_func(df)
        else:
            # Multi-parameter function call
            return original_func(df, **params)

    # Add methods to access the original function and its source
    def get_source():
        """
        Get the source code of the original transform function.

        Returns:
            str: Source code of the original function
        """
        import inspect

        return inspect.getsource(original_func)

    def get_original_function():
        """
        Get the original transform function (unwrapped).

        Returns:
            Callable: The original transform function
        """
        return original_func

    # Attach methods to the wrapper function
    transform_wrapper.get_source = get_source
    transform_wrapper.get_original_function = get_original_function

    return transform_wrapper


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


def _load_schema_constraint(
    name: str,
    version: int | str,
) -> PartialSchemaConstraint | None:
    """
    Load schema constraint from MLflow model metadata.

    Args:
        name: Model name in registry
        version: Model version

    Returns:
        PartialSchemaConstraint if found, None otherwise
    """
    try:
        client = mlflow.tracking.MlflowClient()
        model_version = client.get_model_version(name, str(version))
        run = client.get_run(model_version.run_id)

        constraint_json = run.data.tags.get("schema_constraint")
        if constraint_json:
            return PartialSchemaConstraint.from_json(constraint_json)

    except Exception as e:
        logger.warning(
            "Could not load schema constraint for %s v%s, proceeding without validation: %s",
            name,
            version,
            e,
        )

    return None
