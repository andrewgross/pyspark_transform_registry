"""
PySpark Transform Registry

A simplified library for registering and loading PySpark transform functions
using MLflow's model registry. Supports both single-parameter and multi-parameter
functions with automatic dependency detection and signature inference.
"""

# Import the new simplified API
from .core import get_latest_function_version, load_function, register_function

# Import utility functions for backwards compatibility
from .metadata import _resolve_fully_qualified_name

# Keep model wrapper for advanced usage
from .model_wrapper import PySparkTransformModel


__version__ = "0.1.0"

__all__ = [
    # New API
    "register_function",
    "load_function",
    "get_latest_function_version",
    "PySparkTransformModel",
    "_resolve_fully_qualified_name",
    # Backwards compatibility
    "find_transform_versions",
]
