"""
PySpark Transform Registry

A simplified library for registering and loading PySpark transform functions
using MLflow's model registry. Supports both single-parameter and multi-parameter
functions with automatic dependency detection and signature inference.
"""

import warnings
from typing import Any
from collections.abc import Callable

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


# Backwards compatibility aliases with deprecation warnings
def log_transform_function(*args, **kwargs) -> Any:
    """
    DEPRECATED: Use register_function() instead.

    This function will be removed in a future version.
    """
    warnings.warn(
        "log_transform_function is deprecated. Use register_function() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return register_function(*args, **kwargs)


def load_transform_function(*args, **kwargs) -> Callable:
    """
    DEPRECATED: Use load_function() instead.

    This function will be removed in a future version.
    """
    warnings.warn(
        "load_transform_function is deprecated. Use load_function() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return load_function(*args, **kwargs)


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


__version__ = "1.0.0"

__all__ = [
    # New API
    "register_function",
    "load_function",
    "PySparkTransformModel",
    # Backwards compatibility
    "log_transform_function",
    "load_transform_function",
    "find_transform_versions",
    "_resolve_fully_qualified_name",
    "validate_transform_input",
]
