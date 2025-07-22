"""
Static analysis module for PySpark transform functions.

This module provides LibCST-based static analysis to infer schema constraints
from PySpark transform function source code without executing the functions.
"""

from .analyzer import analyze_function
from .column_analyzer import ColumnAnalyzer, find_column_references
from .operation_analyzer import OperationAnalyzer, find_dataframe_operations
from .schema_inference import (
    ConstraintGenerator,
    generate_constraint,
    generate_constraint_from_function,
)
from .type_inference import TypeInferenceEngine, infer_expression_type

__all__ = [
    # Main analysis function
    "analyze_function",
    # Analysis components
    "ColumnAnalyzer",
    "OperationAnalyzer",
    "TypeInferenceEngine",
    "ConstraintGenerator",
    # Utility functions
    "find_column_references",
    "find_dataframe_operations",
    "infer_expression_type",
    "generate_constraint_from_function",
    "generate_constraint",
]
