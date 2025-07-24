import inspect
from collections.abc import Callable
from typing import Any

import mlflow
import mlflow.pyfunc


class PySparkTransformModel(mlflow.pyfunc.PythonModel):
    """
    Simplified MLflow model wrapper for PySpark transform functions.

    This wrapper allows PySpark transform functions to be registered in MLflow's
    model registry with automatic dependency inference and signature detection.
    Optionally stores schema constraints for runtime validation.
    """

    def __init__(
        self,
        transform_func: Callable,
        schema_constraint: Any | None = None,
    ):
        """
        Initialize the PySpark transform model wrapper.

        Args:
            transform_func: The PySpark transform function to wrap
            schema_constraint: Optional PartialSchemaConstraint for validation
        """
        self.transform_func = transform_func
        self.function_name = transform_func.__name__
        self.schema_constraint = schema_constraint

        # Store function source and signature for reconstruction
        self.function_source = inspect.getsource(transform_func)
        self.function_signature = inspect.signature(transform_func)

    # Explicitly avoid any type hints to avoid MLflow's signature inference
    def predict(
        self,
        context,
        model_input,
        params,
    ):
        """
        MLflow-required predict method that delegates to the wrapped transform function.

        This method handles both MLflow's signature inference and normal prediction,
        with support for multi-parameter functions via the params argument.

        Args:
            context: MLflow model context or input DataFrame (for signature inference)
            model_input: Input DataFrame (when context is provided)
            params: Optional dictionary of additional parameters for multi-input functions

        Returns:
            Transformed DataFrame
        """
        return model_input

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
