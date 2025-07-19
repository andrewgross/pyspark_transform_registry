import inspect
from typing import Any, Callable

import mlflow
import mlflow.pyfunc


class PySparkTransformModel(mlflow.pyfunc.PythonModel):
    """
    Simplified MLflow model wrapper for PySpark transform functions.

    This wrapper allows PySpark transform functions to be registered in MLflow's
    model registry with automatic dependency inference and signature detection.
    """

    def __init__(self, transform_func: Callable):
        """
        Initialize the PySpark transform model wrapper.

        Args:
            transform_func: The PySpark transform function to wrap
        """
        self.transform_func = transform_func
        self.function_name = transform_func.__name__

        # Store function source and signature for reconstruction
        self.function_source = inspect.getsource(transform_func)
        self.function_signature = inspect.signature(transform_func)

    def predict(self, context, model_input=None):
        """
        MLflow-required predict method that delegates to the wrapped transform function.

        This method handles both MLflow's signature inference and normal prediction.

        Args:
            context: MLflow model context or input DataFrame (for signature inference)
            model_input: Input DataFrame (when context is provided)

        Returns:
            Transformed DataFrame
        """
        # Handle signature inference case (single argument)
        if model_input is None:
            return self.transform_func(context)
        # Handle normal prediction case (context, model_input)
        else:
            return self.transform_func(model_input)

    def get_transform_function(self) -> Callable:
        """
        Get the original transform function to preserve existing API.

        Returns:
            The original PySpark transform function
        """
        return self.transform_func

    def get_function_name(self) -> str:
        """Get the name of the wrapped function."""
        return self.function_name

    def get_metadata(self) -> dict[str, Any]:
        """Get metadata about the wrapped function."""
        return {
            "function_name": self.function_name,
            "signature": str(self.function_signature),
            "docstring": self.transform_func.__doc__,
        }

    def get_signature(self) -> inspect.Signature:
        """Get the signature of the wrapped function."""
        return self.function_signature


def create_transform_model(func: Callable) -> PySparkTransformModel:
    """
    Create a PySpark transform model wrapper.

    Args:
        func: The PySpark transform function to wrap

    Returns:
        PySparkTransformModel instance ready for MLflow registration
    """
    return PySparkTransformModel(func)


def extract_transform_function(model: PySparkTransformModel) -> Callable:
    """
    Extract the original transform function from a model wrapper.

    Args:
        model: PySparkTransformModel instance

    Returns:
        Original PySpark transform function
    """
    return model.get_transform_function()
