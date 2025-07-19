"""
PySpark Transform Registry

A simplified library for registering and loading PySpark transform functions
using MLflow's model registry.
"""

# Import the new simplified API
from .core import (
    register_function,
    load_function,
    list_registered_functions,
)

# Keep model wrapper for advanced usage
from .model_wrapper import PySparkTransformModel

__version__ = "1.0.0"

__all__ = [
    "register_function",
    "load_function",
    "list_registered_functions",
    "PySparkTransformModel",
]
