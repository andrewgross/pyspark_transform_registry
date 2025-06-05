import importlib.util
import inspect
import json
import os
import pydoc
import tempfile
import typing
from datetime import datetime
from typing import Callable, Optional

import mlflow
import pyspark
from pyspark.sql import SparkSession

spark = SparkSession.getActiveSession()

def _resolve_fully_qualified_name(obj):
    if obj is None:
        return None
    module = obj.__module__
    qualname = getattr(obj, '__qualname__', obj.__name__)
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
        param_info.append({
            "name": name,
            "annotation": _resolve_fully_qualified_name(annotation) if annotation else None,
            "default": param.default if param.default != inspect._empty else None
        })
    return_type = hints.get("return")
    return_annot = _resolve_fully_qualified_name(return_type) if return_type else None
    doc = inspect.getdoc(func) or ""
    return param_info, return_annot, doc

def _wrap_function_source(name: str, source: str, doc: str, param_info, return_type: Optional[str]):
    """
    Creates a wrapped version of the function's source code with parameter and return type metadata
    and docstring embedded as a header comment.
    """
    header = f"""
Auto-logged transform function: {name}

Args:
"""
    for p in param_info:
        annotation = f" ({p['annotation']})" if p['annotation'] else ""
        default = f", default={p['default']}" if p['default'] is not None else ""
        header += f"  - {p['name']}{annotation}{default}\n"
    header += f"\nReturns: {return_type or 'unspecified'}\n\n{doc}\n"
    return f"{header}{source}"

def log_transform_function(func: Callable, name: str, artifact_path: str = "transform_code", as_text: bool = True):
    """
    Logs a PySpark transform function's source code to MLflow, with metadata and docstring header.
    """
    source = inspect.getsource(func)
    param_info, return_type, doc = _get_function_metadata(func)
    wrapped_source = _wrap_function_source(name, source, doc, param_info, return_type)
    filename = f"{name}.py"

    if as_text:
        mlflow.log_text(wrapped_source, f"{artifact_path}/{filename}")
    else:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, filename)
            with open(path, "w") as f:
                f.write(wrapped_source)
            mlflow.log_artifact(path, artifact_path)

    run_id = mlflow.active_run().info.run_id
    track_transform_version(name, run_id, artifact_path, return_type, doc, param_info)

def load_transform_function(run_id: str, name: str, artifact_path: str = "transform_code") -> Callable:
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

def track_transform_version(name: str, run_id: str, artifact_path: str, return_type: Optional[str], doc: str, param_info) -> pyspark.sql.DataFrame:
    """
    Tracks transform metadata in a Delta table for discoverability and version control.
    """
    timestamp = datetime.utcnow().isoformat()
    df = spark.createDataFrame([{
        "function_name": name,
        "run_id": run_id,
        "artifact_path": artifact_path,
        "return_type": return_type,
        "param_info_json": json.dumps(param_info),
        "docstring": doc,
        "timestamp": timestamp
    }])
    return df

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
    actual_type = type(input_obj)
    if expected_type is None:
        return True

    resolved = pydoc.locate(f"{expected_type.__module__}.{expected_type.__qualname__}")
    return isinstance(input_obj, resolved) if resolved else False
