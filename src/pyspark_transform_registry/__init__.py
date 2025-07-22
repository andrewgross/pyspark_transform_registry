"""
PySpark Transform Registry

A simplified library for registering and loading PySpark transform functions
using MLflow's model registry. Supports both single-parameter and multi-parameter
functions with automatic dependency detection and signature inference.
"""

import warnings
from typing import Any

# Import the new simplified API
from .core import (
    register_function,
    load_function,
)

# Keep model wrapper for advanced usage
from .model_wrapper import PySparkTransformModel

# Import utility functions for backwards compatibility
from .metadata import _resolve_fully_qualified_name
from .validation import validate_transform_input


def find_transform_versions(*args, **kwargs) -> Any:
    """
    DEPRECATED: This function has been removed.

    Use MLflow's native model registry APIs directly for model discovery.
    """
    warnings.warn(
        "find_transform_versions has been removed. Use MLflow's model registry APIs directly.",
        DeprecationWarning,
        stacklevel=2,
    )
    raise NotImplementedError(
        "find_transform_versions has been removed. Use MLflow's model registry APIs directly.",
    )


__version__ = "0.1.0"

__all__ = [
    # New API
    "register_function",
    "load_function",
    "PySparkTransformModel",
    "_resolve_fully_qualified_name",
    "validate_transform_input",
    # Backwards compatibility
    "find_transform_versions",
]
