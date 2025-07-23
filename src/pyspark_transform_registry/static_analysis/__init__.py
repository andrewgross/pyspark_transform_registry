"""
Static analysis module for PySpark transform functions.

This module provides AST-based static analysis to infer schema constraints
from PySpark transform function source code without executing the functions.
"""

from .analyzer import analyze_function
from .ast_analyzers import (
    ASTColumnAnalyzer,
    ASTOperationAnalyzer,
    ASTTypeInferenceEngine,
    ColumnReference,
    DataFrameOperation,
    DataFrameVariableTracker,
)
from .schema_inference import (
    ConstraintGenerator,
    generate_constraint,
    generate_constraint_from_function,
)

__all__ = [
    # Main analysis function
    "analyze_function",
    # AST-based analysis components
    "ASTColumnAnalyzer",
    "ASTOperationAnalyzer",
    "ASTTypeInferenceEngine",
    "DataFrameVariableTracker",
    "ColumnReference",
    "DataFrameOperation",
    "ConstraintGenerator",
    # Utility functions
    "generate_constraint_from_function",
    "generate_constraint",
]
