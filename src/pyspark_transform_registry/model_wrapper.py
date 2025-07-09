import inspect
from typing import Any, Callable, Optional

import mlflow
import mlflow.pyfunc
from pyspark.sql import DataFrame


class PySparkTransformModel(mlflow.pyfunc.PythonModel):
    """
    MLflow model wrapper for PySpark transform functions.

    This wrapper allows PySpark transform functions to be registered in MLflow's
    model registry while preserving the original function interface and enabling
    automatic input/output validation through MLflow signatures.
    """

    def __init__(
        self,
        transform_func: Callable,
        function_name: str,
        metadata: dict[str, Any],
    ):
        """
        Initialize the PySpark transform model wrapper.

        Args:
            transform_func: The PySpark transform function to wrap
            function_name: Name of the function for identification
            metadata: Additional metadata about the function (params, return type, etc.)
        """
        self.transform_func = transform_func
        self.function_name = function_name
        self.metadata = metadata

        # Store function source and signature for reconstruction
        self.function_source = inspect.getsource(transform_func)
        self.function_signature = inspect.signature(transform_func)

    def predict(
        self,
        context: mlflow.pyfunc.PythonModelContext,
        model_input: DataFrame,
    ) -> DataFrame:
        """
        MLflow-required predict method that delegates to the wrapped transform function.

        Args:
            context: MLflow model context (unused for transforms)
            model_input: Input DataFrame to transform

        Returns:
            Transformed DataFrame
        """
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
        return self.metadata

    def get_signature(self) -> inspect.Signature:
        """Get the signature of the wrapped function."""
        return self.function_signature


def create_transform_model(
    func: Callable,
    name: str,
    metadata: Optional[dict[str, Any]] = None,
) -> PySparkTransformModel:
    """
    Create a PySpark transform model wrapper.

    Args:
        func: The PySpark transform function to wrap
        name: Name for the transform function
        metadata: Optional metadata about the function

    Returns:
        PySparkTransformModel instance ready for MLflow registration
    """
    if metadata is None:
        metadata = {}

    return PySparkTransformModel(func, name, metadata)


def extract_transform_function(model: PySparkTransformModel) -> Callable:
    """
    Extract the original transform function from a model wrapper.

    Args:
        model: PySparkTransformModel instance

    Returns:
        Original PySpark transform function
    """
    return model.get_transform_function()
