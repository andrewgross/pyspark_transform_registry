import datetime
import inspect
import json
from enum import Enum
from typing import Callable, Optional

import mlflow
import mlflow.pyfunc
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import Column

from .metadata import _get_function_metadata
from .model_wrapper import PySparkTransformModel, create_transform_model
from .signature_inference import infer_pyspark_signature, create_signature_from_examples
from .versioning import generate_next_version, parse_semantic_version


class TransformType(Enum):
    """Types of PySpark transform functions."""

    DATAFRAME_TRANSFORM = "dataframe_transform"  # df -> df
    COLUMN_EXPRESSION = "column_expression"  # * -> col
    CUSTOM = "custom"  # other patterns


def _detect_transform_type(func: Callable) -> TransformType:
    """Automatically detect the type of transform function."""
    try:
        sig = inspect.signature(func)
        params = list(sig.parameters.values())

        # DataFrame transform: first param is DataFrame, returns DataFrame
        if (
            params
            and params[0].annotation == DataFrame
            and sig.return_annotation == DataFrame
        ):
            return TransformType.DATAFRAME_TRANSFORM

        # Column expression: returns Column (regardless of input types)
        elif sig.return_annotation == Column:
            return TransformType.COLUMN_EXPRESSION

        else:
            return TransformType.CUSTOM

    except Exception:
        return TransformType.CUSTOM


def _suggest_calling_convention(transform_type: TransformType) -> str:
    """Suggest calling convention based on transform type."""
    if transform_type == TransformType.DATAFRAME_TRANSFORM:
        return "df.transform(func, **kwargs) or func(df, **kwargs)"
    elif transform_type == TransformType.COLUMN_EXPRESSION:
        return "df.withColumn('col_name', func(col('input')))"
    else:
        return "See function documentation for usage"


spark = SparkSession.getActiveSession()


def log_transform_function(
    func: Callable,
    *,
    name: Optional[str] = None,
    input_example: Optional[DataFrame] = None,
    output_example: Optional[DataFrame] = None,
    version: Optional[str] = None,
    version_bump: Optional[str] = None,
    registered_model_name: Optional[str] = None,
    extra_pip_requirements: Optional[list[str]] = None,
    code_paths: Optional[list[str]] = None,
    validate_dependencies: bool = True,
    auto_detect_requirements: bool = True,
) -> str:
    """
    Registers a PySpark transform function in MLflow's model registry with automatic validation.

    Args:
        func: The PySpark transform function to register
        name: Name for the transform function (defaults to func.__name__)
        input_example: Optional example input DataFrame for schema inference
        output_example: Optional example output DataFrame for schema inference
        version: Optional explicit semantic version (e.g., "1.2.0")
        version_bump: Optional explicit version bump type ("major", "minor", "patch")
        registered_model_name: Optional name for the registered model (defaults to name)
        extra_pip_requirements: Additional pip requirements to include
        code_paths: Local code paths to bundle with the transform
        validate_dependencies: Whether to validate function dependencies and warn about issues
        auto_detect_requirements: Whether to automatically detect minimal requirements

    Returns:
        Model URI of the registered model
    """
    # Use function name if no name provided
    if name is None:
        name = func.__name__

    # Analyze dependencies if requested
    final_pip_requirements = None
    final_code_paths = None

    if auto_detect_requirements or validate_dependencies:
        from .requirements_analysis import (
            create_minimal_requirements,
            validate_function_safety,
            DependencyAnalyzer,
        )

        if validate_dependencies:
            # Validate function safety and warn about potential issues
            analyzer = DependencyAnalyzer()
            validation = validate_function_safety(func, analyzer)

            if validation["warnings"]:
                print(f"⚠️  Dependency warnings for function '{name}':")
                for warning in validation["warnings"]:
                    print(f"   - {warning}")

            if validation["errors"]:
                print(f"❌ Dependency errors for function '{name}':")
                for error in validation["errors"]:
                    print(f"   - {error}")
                raise ValueError(
                    f"Function '{name}' has dependency issues that prevent safe bundling",
                )

        if auto_detect_requirements:
            # Create minimal requirements
            requirements_spec = create_minimal_requirements(
                func,
                extra_requirements=extra_pip_requirements,
                code_paths=code_paths,
            )

            final_pip_requirements = requirements_spec["pip_requirements"]
            final_code_paths = requirements_spec["code_paths"]

            if requirements_spec["warnings"]:
                print(f"ℹ️  Detected dependencies for function '{name}':")
                for warning in requirements_spec["warnings"]:
                    print(f"   - {warning}")

            print(f"📦 Minimal requirements detected: {final_pip_requirements}")
            if final_code_paths:
                print(f"📁 Code paths to bundle: {final_code_paths}")
    else:
        # Use manual requirements if provided
        final_pip_requirements = extra_pip_requirements
        final_code_paths = code_paths

    # Extract function metadata
    param_info, return_type, doc = _get_function_metadata(func)

    # Detect transform type for metadata
    transform_type = _detect_transform_type(func)
    calling_convention = _suggest_calling_convention(transform_type)

    # Create model wrapper
    metadata = {
        "param_info": param_info,
        "return_type": return_type,
        "docstring": doc,
        "function_source": inspect.getsource(func),
        "transform_type": transform_type.value,
        "calling_convention": calling_convention,
    }

    model = create_transform_model(func, name, metadata)

    # Convert PySpark DataFrame to pandas for MLflow compatibility
    pandas_input_example = None
    if input_example is not None:
        try:
            pandas_input_example = input_example.toPandas()
        except Exception as e:
            print(f"Warning: Could not convert input example to pandas: {e}")

    # Infer signature from examples or function signature
    signature = None
    if input_example is not None and output_example is not None:
        signature = create_signature_from_examples(input_example, output_example)
    elif input_example is not None:
        # Try to run the function to get output example
        try:
            output_example = func(input_example)
            signature = create_signature_from_examples(input_example, output_example)
        except Exception as e:
            print(f"Warning: Could not run function to infer output schema: {e}")
            signature = infer_pyspark_signature(func, input_example)
    else:
        signature = infer_pyspark_signature(func)

    # Handle versioning
    if version is not None:
        # Validate and use explicit version
        semantic_version = parse_semantic_version(version)
    else:
        # Auto-generate version based on interface changes
        semantic_version = generate_next_version(name, func, version_bump)

    # Create model info with tags for semantic versioning
    model_info = mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=model,
        signature=signature,
        input_example=pandas_input_example,
        registered_model_name=registered_model_name or name,
        extra_pip_requirements=final_pip_requirements,
        code_paths=final_code_paths,
        metadata={
            "transform_type": "pyspark",
            "transform_name": name,
            "semantic_version": str(semantic_version),
            "major_version": str(semantic_version.major),
            "minor_version": str(semantic_version.minor),
            "patch_version": str(semantic_version.patch),
            "param_info": json.dumps(param_info),
            "return_type": return_type or "unspecified",
            "docstring": doc,
            "pyspark_transform_type": transform_type.value,
            "calling_convention": calling_convention,
            "timestamp": str(datetime.datetime.now(datetime.timezone.utc).timestamp()),
        },
    )

    # Set tags on the run for backwards compatibility
    mlflow.set_tag("transform_type", "pyspark")
    mlflow.set_tag("transform_name", name)
    mlflow.set_tag("semantic_version", str(semantic_version))

    return model_info.model_uri


def log_transform_cluster(
    functions: list[Callable],
    cluster_name: str,
    *,
    input_example: Optional[DataFrame] = None,
    output_example: Optional[DataFrame] = None,
    version: Optional[str] = None,
    version_bump: Optional[str] = None,
    registered_model_name: Optional[str] = None,
    extra_pip_requirements: Optional[list[str]] = None,
    code_paths: Optional[list[str]] = None,
    validate_dependencies: bool = True,
) -> str:
    """
    Log a cluster of related functions that should be bundled together.

    This is useful when you have multiple functions that call each other
    or share common dependencies, ensuring they're all available when
    any transform from the cluster is loaded.

    Args:
        functions: List of functions to bundle together
        cluster_name: Name for the function cluster
        input_example: Optional example input DataFrame for schema inference
        output_example: Optional example output DataFrame for schema inference
        version: Optional explicit semantic version
        version_bump: Optional explicit version bump type
        registered_model_name: Optional name for the registered model
        extra_pip_requirements: Additional pip requirements to include
        code_paths: Local code paths to bundle with the cluster
        validate_dependencies: Whether to validate function dependencies

    Returns:
        Model URI of the registered cluster
    """
    if not functions:
        raise ValueError("At least one function must be provided")

    # Create a function cluster and analyze dependencies
    from .requirements_analysis import FunctionCluster, DependencyAnalyzer

    cluster = FunctionCluster(cluster_name)
    for func in functions:
        cluster.add_function(func)

    if code_paths:
        for path in code_paths:
            cluster.add_local_code_path(path)

    # Analyze cluster dependencies
    analyzer = DependencyAnalyzer()
    cluster_deps = cluster.analyze_cluster_dependencies(analyzer)

    if validate_dependencies:
        # Validate each function in the cluster
        all_warnings = []
        for func in functions:
            from .requirements_analysis import validate_function_safety

            validation = validate_function_safety(func, analyzer)

            if validation["warnings"]:
                all_warnings.extend(validation["warnings"])

            if validation["errors"]:
                print(f"❌ Dependency errors for function '{func.__name__}':")
                for error in validation["errors"]:
                    print(f"   - {error}")
                raise ValueError(
                    f"Function cluster '{cluster_name}' has dependency issues",
                )

        if all_warnings:
            print(f"⚠️  Dependency warnings for cluster '{cluster_name}':")
            for warning in set(all_warnings):  # Remove duplicates
                print(f"   - {warning}")

    # Combine requirements
    final_pip_requirements = cluster_deps["pip_requirements"]
    if extra_pip_requirements:
        final_pip_requirements.extend(extra_pip_requirements)

    final_code_paths = cluster_deps["code_paths"]

    print(f"📦 Cluster '{cluster_name}' requirements: {final_pip_requirements}")
    if final_code_paths:
        print(f"📁 Cluster code paths: {final_code_paths}")

    # Create a wrapper function that contains all functions
    def cluster_wrapper_factory():
        # Create a namespace with all functions
        namespace = {}
        for func in functions:
            namespace[func.__name__] = func

        # Create the main cluster function
        def cluster_transform(df: DataFrame, function_name: str, **kwargs) -> DataFrame:
            """
            Execute a function from the cluster by name.

            Args:
                df: Input DataFrame
                function_name: Name of function to execute
                **kwargs: Arguments to pass to the function

            Returns:
                Transformed DataFrame
            """
            if function_name not in namespace:
                available = list(namespace.keys())
                raise ValueError(
                    f"Function '{function_name}' not found in cluster. Available: {available}",
                )

            func = namespace[function_name]
            return func(df, **kwargs)

        # Add namespace to the function for access
        setattr(cluster_transform, "_cluster_functions", namespace)
        setattr(cluster_transform, "_cluster_name", cluster_name)

        return cluster_transform

    cluster_function = cluster_wrapper_factory()

    # Log the cluster using the regular log_transform_function
    return log_transform_function(
        cluster_function,
        name=cluster_name,
        input_example=input_example,
        output_example=output_example,
        version=version,
        version_bump=version_bump,
        registered_model_name=registered_model_name,
        extra_pip_requirements=final_pip_requirements,
        code_paths=final_code_paths,
        validate_dependencies=False,  # Already validated above
        auto_detect_requirements=False,  # Already analyzed above
    )


def load_transform_function(
    model_name: str,
    version: Optional[str] = None,
    stage: Optional[str] = None,
    alias: Optional[str] = None,
    run_id: Optional[str] = None,
) -> Callable:
    """
    Loads a transform function from MLflow's model registry.

    Args:
        model_name: Name of the registered model
        version: Optional model version (e.g., "1", "2")
        stage: Optional model stage ("Staging", "Production", "Archived")
        alias: Optional model alias ("champion", "challenger", etc.)
        run_id: Optional run ID for backwards compatibility

    Returns:
        The original PySpark transform function
    """
    # Build model URI based on provided parameters
    if run_id is not None:
        # Backwards compatibility: load from run ID
        model_uri = f"runs:/{run_id}/model"
    elif version is not None:
        model_uri = f"models:/{model_name}/{version}"
    elif stage is not None:
        model_uri = f"models:/{model_name}/{stage}"
    elif alias is not None:
        model_uri = f"models:/{model_name}@{alias}"
    else:
        # Default to latest version
        model_uri = f"models:/{model_name}/latest"

    # Load the model
    loaded_model = mlflow.pyfunc.load_model(model_uri)

    # Extract the transform function from the wrapper
    # MLflow wraps our PySparkTransformModel in a _PythonModelPyfuncWrapper
    if hasattr(loaded_model, "_model_impl") and hasattr(
        loaded_model._model_impl,
        "python_model",
    ):
        python_model = loaded_model._model_impl.python_model
        if isinstance(python_model, PySparkTransformModel):
            return python_model.get_transform_function()

    # Fallback: try direct access
    if hasattr(loaded_model, "_model_impl") and isinstance(
        loaded_model._model_impl,
        PySparkTransformModel,
    ):
        return loaded_model._model_impl.get_transform_function()

    raise ValueError(
        f"Loaded model is not a PySparkTransformModel: {type(loaded_model)}, impl: {type(getattr(loaded_model, '_model_impl', None))}",
    )


def find_transform_versions(
    name: Optional[str] = None,
    return_type: Optional[str] = None,
    version_constraint: Optional[str] = None,
    latest_only: bool = False,
) -> list[dict]:
    """
    Find transform versions using MLflow's model registry.

    Args:
        name: Optional filter by transform name
        return_type: Optional filter by return type
        version_constraint: Optional version constraint (e.g., ">=1.0.0,<2.0.0")
        latest_only: If True, return only the latest version

    Returns:
        List of model version dictionaries with metadata
    """
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    results = []

    # Get all registered models
    registered_models = client.search_registered_models()

    for model in registered_models:
        # Filter by name if specified
        if name is not None and model.name != name:
            continue

        # Get all versions of this model
        model_versions = client.search_model_versions(f"name='{model.name}'")

        for version in model_versions:
            # Check if this is a transform model
            run = client.get_run(version.run_id)
            if run.data.tags.get("transform_type") != "pyspark":
                continue

            # Get model metadata
            try:
                # Access model artifact metadata
                model_uri = f"models:/{model.name}/{version.version}"
                loaded_model = mlflow.pyfunc.load_model(model_uri)
                artifact_metadata = loaded_model.metadata.metadata or {}

                # Filter by return type if specified
                if return_type is not None:
                    model_return_type = artifact_metadata.get(
                        "return_type",
                        run.data.tags.get("return_type"),
                    )
                    if model_return_type != return_type:
                        continue

                # Parse semantic version for constraint checking
                semantic_version_str = artifact_metadata.get(
                    "semantic_version",
                    run.data.tags.get("semantic_version"),
                )
                if semantic_version_str:
                    try:
                        semantic_version = parse_semantic_version(semantic_version_str)

                        # Apply version constraint if specified
                        if version_constraint is not None:
                            from .versioning import satisfies_version_constraint

                            if not satisfies_version_constraint(
                                semantic_version,
                                version_constraint,
                            ):
                                continue

                        # Add to results
                        results.append(
                            {
                                "model_name": model.name,
                                "model_version": version.version,
                                "semantic_version": semantic_version,
                                "stage": version.current_stage,
                                "run_id": version.run_id,
                                "creation_timestamp": version.creation_timestamp,
                                "metadata": {
                                    "transform_name": artifact_metadata.get(
                                        "transform_name",
                                        run.data.tags.get("transform_name"),
                                    ),
                                    "return_type": artifact_metadata.get(
                                        "return_type",
                                        run.data.tags.get("return_type"),
                                    ),
                                    "docstring": artifact_metadata.get(
                                        "docstring",
                                        run.data.tags.get("docstring", ""),
                                    ),
                                    "param_info": artifact_metadata.get(
                                        "param_info",
                                        run.data.tags.get("param_info", "[]"),
                                    ),
                                },
                            },
                        )
                    except ValueError:
                        # Skip invalid version formats
                        continue
            except Exception as e:
                # Skip models that can't be loaded
                print(
                    f"Warning: Could not load model {model.name} version {version.version}: {e}",
                )
                continue

    # Sort by semantic version (descending)
    results.sort(key=lambda x: x["semantic_version"], reverse=True)

    # Return only latest if requested
    if latest_only and results:
        return [results[0]]

    return results


def load_transform_function_by_name(
    name: str,
    version_constraint: Optional[str] = None,
    latest_only: bool = True,
) -> Callable:
    """
    Convenience function to load a transform function by name.

    Args:
        name: Name of the transform function
        version_constraint: Optional version constraint
        latest_only: If True, load only the latest matching version

    Returns:
        The loaded transform function
    """
    versions = find_transform_versions(
        name=name,
        version_constraint=version_constraint,
        latest_only=latest_only,
    )

    if not versions:
        raise ValueError(f"No transform versions found for name '{name}'")

    # Get the first (latest) version
    latest_version = versions[0]
    model_name = latest_version["model_name"]
    model_version = latest_version["model_version"]

    return load_transform_function(model_name, version=model_version)


# Backwards compatibility function for existing API
def load_transform_function_from_run(
    run_id: str,
    name: str,
    artifact_path: str = "transform_code",
) -> Callable:
    """
    Backwards compatibility function for loading from run ID.

    Args:
        run_id: MLflow run ID
        name: Transform function name
        artifact_path: Legacy parameter (ignored)

    Returns:
        The loaded transform function
    """
    return load_transform_function(model_name=name, run_id=run_id)
