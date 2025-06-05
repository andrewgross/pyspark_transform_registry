import datetime
import importlib.util
import inspect
import json
import os
import pydoc
import tempfile
import typing
from typing import Callable, Optional

import mlflow
from pyspark.sql import SparkSession

spark = SparkSession.getActiveSession()


def _resolve_fully_qualified_name(obj):
    if obj is None:
        return None
    module = obj.__module__
    qualname = getattr(obj, "__qualname__", obj.__name__)
    return f"{module}.{qualname}"


def _get_function_metadata(func: Callable):
    """
    Extracts parameter information, return type annotation, and docstring from a function.
    """
    sig = inspect.signature(func)
    hints = typing.get_type_hints(func)
    param_info = []
    for name, param in sig.parameters.items():
        annotation = hints.get(name)
        param_info.append(
            {
                "name": name,
                "annotation": _resolve_fully_qualified_name(annotation)
                if annotation
                else None,
                "default": param.default if param.default != inspect._empty else None,
            },
        )
    return_type = hints.get("return")
    return_annot = _resolve_fully_qualified_name(return_type) if return_type else None
    doc = inspect.getdoc(func) or ""
    return param_info, return_annot, doc


def _wrap_function_source(
    name: str,
    source: str,
    doc: str,
    param_info,
    return_type: Optional[str],
):
    """
    Creates a wrapped version of the function's source code with parameter and return type metadata
    and docstring embedded as a header comment.
    """
    header = f"""
Auto-logged transform function: {name}

Args:
"""
    for p in param_info:
        annotation = f" ({p['annotation']})" if p["annotation"] else ""
        default = f", default={p['default']}" if p["default"] is not None else ""
        header += f"  - {p['name']}{annotation}{default}\n"
    header += f"\nReturns: {return_type or 'unspecified'}\n\n{doc}\n"
    return f"{header}{source}"


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
    if name:
        filter_string += f" AND params.transform_name = '{name}'"
    if return_type:
        filter_string += f" AND params.return_type = '{return_type}'"

    return mlflow.search_runs(filter_string=filter_string)


def validate_transform_input(func: Callable, input_obj) -> bool:
    """
    Validates that the first argument's type of a transform function matches the input object's type.
    """
    sig = inspect.signature(func)
    params = list(sig.parameters.values())
    if not params:
        return True  # no input to validate

    first_param = params[0].name
    hints = typing.get_type_hints(func)
    expected_type = hints.get(first_param)
    if expected_type is None:
        return True

    resolved = pydoc.locate(f"{expected_type.__module__}.{expected_type.__qualname__}")
    return isinstance(input_obj, resolved) if resolved else False
