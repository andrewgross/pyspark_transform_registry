"""
Main static analysis orchestrator.

This module coordinates all the analysis components to generate complete
schema constraints from PySpark transform function source code.
"""

import inspect
from typing import Callable
import libcst as cst

from ..schema_constraints import PartialSchemaConstraint
from .column_analyzer import ColumnAnalyzer
from .operation_analyzer import OperationAnalyzer
from .type_inference import TypeInferenceEngine
from .schema_inference import ConstraintGenerator


def analyze_function(func: Callable) -> PartialSchemaConstraint:
    """
    Analyze a PySpark transform function to generate schema constraints.

    Args:
        func: The function to analyze

    Returns:
        PartialSchemaConstraint with inferred requirements and transformations
    """
    try:
        # Extract function source code
        source = inspect.getsource(func)

        # Parse with LibCST
        try:
            tree = cst.parse_module(source)
        except Exception as e:
            # If we can't parse the source, return empty constraint
            constraint = PartialSchemaConstraint(
                analysis_method="static_analysis",
            )
            constraint.add_warning(f"Could not parse function source: {e}")
            return constraint

        # Initialize analyzers
        column_analyzer = ColumnAnalyzer()
        operation_analyzer = OperationAnalyzer()
        type_engine = TypeInferenceEngine()
        constraint_generator = ConstraintGenerator()

        # Visit the tree with our analyzers
        wrapper = AnalysisWrapper(column_analyzer, operation_analyzer, type_engine)
        tree.visit(wrapper)

        # Generate final constraint
        constraint = constraint_generator.generate_constraint(
            operations=operation_analyzer.operations,
            column_references=column_analyzer.column_references,
            type_info=type_engine.type_mappings,
            source_analysis=wrapper.get_analysis_summary(),
        )

        return constraint

    except Exception as e:
        # Fallback for any unexpected errors
        constraint = PartialSchemaConstraint(
            analysis_method="static_analysis",
        )
        constraint.add_warning(f"Analysis failed: {e}")
        return constraint


class AnalysisWrapper(cst.CSTVisitor):
    """
    LibCST visitor that coordinates multiple analysis components.

    This wrapper visits the CST and delegates to specific analyzers
    while tracking overall analysis quality.
    """

    def __init__(
        self,
        column_analyzer: ColumnAnalyzer,
        operation_analyzer: OperationAnalyzer,
        type_engine: TypeInferenceEngine,
    ):
        self.column_analyzer = column_analyzer
        self.operation_analyzer = operation_analyzer
        self.type_engine = type_engine

        # Track analysis quality indicators
        self.udf_count = 0
        self.dynamic_operations = 0
        self.complex_expressions = 0
        self.unparseable_expressions = 0

    def visit_Call(self, node: cst.Call) -> None:
        """Visit function calls - delegate to appropriate analyzers."""
        # Let each analyzer process the call
        self.column_analyzer.visit_call(node)
        self.operation_analyzer.visit_call(node)
        self.type_engine.visit_call(node)

        # Track UDF usage
        if self._is_udf_decorator(node):
            self.udf_count += 1

    def visit_Attribute(self, node: cst.Attribute) -> None:
        """Visit attribute access - mainly for df.column patterns."""
        self.column_analyzer.visit_attribute(node)

    def visit_Subscript(self, node: cst.Subscript) -> None:
        """Visit subscript access - for df["column"] patterns."""
        self.column_analyzer.visit_subscript(node)

    def visit_For(self, node: cst.For) -> None:
        """Visit for loops - may indicate dynamic operations."""
        # Check if this looks like dynamic column operations
        if self._is_dynamic_column_operation(node):
            self.dynamic_operations += 1

    def visit_If(self, node: cst.If) -> None:
        """Visit if statements - may indicate complex conditional logic."""
        # Track complex conditional expressions
        if self._is_complex_condition(node):
            self.complex_expressions += 1

    def _is_udf_decorator(self, node: cst.Call) -> bool:
        """Check if a call represents a UDF decorator."""
        # Look for @F.udf() patterns
        if isinstance(node.func, cst.Attribute):
            if (
                isinstance(node.func.value, cst.Name)
                and node.func.value.value == "F"
                and node.func.attr.value == "udf"
            ):
                return True
        return False

    def _is_dynamic_column_operation(self, node: cst.For) -> bool:
        """Check if a for loop represents dynamic column operations."""
        # This is a heuristic - look for patterns like:
        # for col in columns: df.withColumn(f"...", ...)

        # Currently assumes any for loop might be dynamic column creation
        # This is conservative but prevents missed dynamic operations
        # Future enhancement: analyze loop body for actual column operations
        return True

    def _is_complex_condition(self, node: cst.If) -> bool:
        """Check if an if statement has complex conditional logic."""
        # Currently treats all conditional logic as potentially complex
        # This ensures schema constraints reflect possible code paths
        # Future enhancement: analyze condition complexity and simplicity patterns
        return True

    def get_analysis_summary(self) -> dict:
        """Get summary of analysis quality indicators."""
        return {
            "udf_count": self.udf_count,
            "dynamic_operations": self.dynamic_operations,
            "complex_expressions": self.complex_expressions,
            "unparseable_expressions": self.unparseable_expressions,
        }
