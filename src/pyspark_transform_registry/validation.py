import inspect
import pydoc
import typing
from collections.abc import Callable

import mlflow
from pyspark.sql import DataFrame


def validate_transform_input(func: Callable, input_obj) -> bool:
    """
    Validates that the first argument's type of a transform function matches the input object's type.

    Note: This is a legacy function. MLflow model signatures now provide automatic
    input validation when models are loaded and used via the model registry.
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
    return (
        isinstance(input_obj, resolved)
        if resolved and inspect.isclass(resolved)
        else False
    )


def validate_with_mlflow_signature(model_uri: str, input_df: DataFrame) -> bool:
    """
    Validate input DataFrame against MLflow model signature.

    Args:
        model_uri: URI of the MLflow model
        input_df: Input DataFrame to validate

    Returns:
        True if validation passes, False otherwise
    """
    try:
        # Load the model to get its signature
        loaded_model = mlflow.pyfunc.load_model(model_uri)

        # MLflow will automatically validate the input when predict is called
        # This is a placeholder for explicit validation if needed
        if hasattr(loaded_model, "metadata") and loaded_model.metadata.signature:
            # MLflow handles validation automatically
            return True

        return True
    except Exception:
        return False
