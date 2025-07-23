"""
AST-based static analysis orchestrator for PySpark transform functions.

This module coordinates all analysis components to generate complete
schema constraints from PySpark transform function source code using
Python's built-in AST module for robust static analysis.
"""

import ast
import inspect
import textwrap
from collections.abc import Callable
from typing import Any, get_type_hints

from ..schema_constraints import PartialSchemaConstraint
from .ast_analyzers import (
    ASTColumnAnalyzer,
    ASTOperationAnalyzer,
    ASTTypeInferenceEngine,
    DataFrameVariableTracker,
)
from .schema_inference import ConstraintGenerator


def analyze_function(func: Callable) -> PartialSchemaConstraint:
    """
    Analyze a PySpark transform function to generate schema constraints.

    This function uses Python's AST module to parse and analyze the function
    source code, providing robust DataFrame variable detection and type inference.

    Args:
        func: The function to analyze

    Returns:
        PartialSchemaConstraint with inferred requirements and transformations

    Raises:
        ValueError: If function source cannot be parsed
        TypeError: If function doesn't appear to be a DataFrame transform
    """
    # Extract function source code
    try:
        source = inspect.getsource(func)
    except (OSError, TypeError) as e:
        raise ValueError(f"Could not extract function source: {e}")

    # Parse with Python's AST module
    try:
        # Dedent the source code to handle functions defined inside other contexts
        dedented_source = textwrap.dedent(source)
        tree = ast.parse(dedented_source)
    except SyntaxError as e:
        raise ValueError(f"Could not parse function source - syntax error: {e}")

    # Get function type hints for DataFrame parameter detection
    try:
        type_hints = get_type_hints(func)
    except (AttributeError, NameError, TypeError):
        # Type hints may not be available or may reference undefined types
        type_hints = {}

    # Get function signature for parameter analysis
    try:
        signature = inspect.signature(func)
    except (ValueError, TypeError) as e:
        raise ValueError(f"Could not extract function signature: {e}")

    # Initialize the main analyzer
    analyzer = PySparkTransformAnalyzer(func, type_hints, signature)

    # Visit the AST tree to collect analysis data
    analyzer.visit(tree)

    # Generate final constraint from analysis results
    constraint_generator = ConstraintGenerator()
    constraint = constraint_generator.generate_constraint(
        operations=analyzer.operation_analyzer.get_operations(),
        column_references=analyzer.column_analyzer.get_column_references(),
        type_info=analyzer.type_engine.get_type_mappings(),
        source_analysis=analyzer.get_analysis_summary(),
    )

    return constraint


class PySparkTransformAnalyzer(ast.NodeVisitor):
    """
    Main AST visitor that orchestrates analysis of PySpark transform functions.

    This analyzer coordinates multiple specialized analyzers:
    - DataFrameVariableTracker: Detects DataFrame parameters and variables
    - ASTColumnAnalyzer: Analyzes column references and operations
    - ASTOperationAnalyzer: Detects DataFrame method calls and transformations
    - ASTTypeInferenceEngine: Infers types of expressions and columns
    """

    def __init__(
        self,
        func: Callable,
        type_hints: dict[str, Any],
        signature: inspect.Signature,
    ):
        """
        Initialize the analyzer with function metadata.

        Args:
            func: The function being analyzed
            type_hints: Type hints extracted from the function
            signature: Function signature with parameter information
        """
        self.func = func
        self.func_name = func.__name__
        self.type_hints = type_hints
        self.signature = signature

        # Initialize specialized analyzers
        self.df_tracker = DataFrameVariableTracker(type_hints, signature)
        self.column_analyzer = ASTColumnAnalyzer(self.df_tracker)
        self.operation_analyzer = ASTOperationAnalyzer(self.df_tracker)
        self.type_engine = ASTTypeInferenceEngine(self.df_tracker, type_hints)

        # Track analysis quality indicators
        self.udf_count = 0
        self.dynamic_operations = 0
        self.complex_expressions = 0
        self.function_def_found = False
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """
        Visit function definition to analyze parameters and setup DataFrame tracking.

        Args:
            node: AST FunctionDef node
        """
        if node.name == self.func_name:
            self.function_def_found = True

            # Let the DataFrame tracker analyze function parameters
            self.df_tracker.analyze_function_def(node)

        # Let the type engine analyze ALL function definitions for UDF detection
        # This includes nested UDFs defined inside the main transform function
        self.type_engine.analyze_function_def(node)

        # Continue visiting child nodes
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        """
        Visit assignments to track DataFrame variable assignments.

        Args:
            node: AST Assign node
        """
        # Let DataFrame tracker analyze variable assignments
        self.df_tracker.analyze_assignment(node)

        # Let column analyzer track column list assignments
        self.column_analyzer.analyze_assignment(node)

        # Let type engine infer types from assignments
        self.type_engine.analyze_assignment(node)

        # Continue visiting child nodes
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        """
        Visit function calls to detect DataFrame operations and UDF usage.

        Args:
            node: AST Call node
        """
        # Analyze with specialized analyzers
        self.column_analyzer.analyze_call(node)
        self.operation_analyzer.analyze_call(node)
        self.type_engine.analyze_call(node)

        # Check for UDF usage patterns
        if self._is_udf_call(node):
            self.udf_count += 1

        # Continue visiting child nodes
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        """
        Visit attribute access for column references and method chaining.

        Args:
            node: AST Attribute node
        """
        # Analyze column references (df.column_name patterns)
        self.column_analyzer.analyze_attribute(node)

        # Continue visiting child nodes
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        """
        Visit subscript access for bracket notation column references.

        Args:
            node: AST Subscript node
        """
        # Analyze column references (df["column_name"] patterns)
        self.column_analyzer.analyze_subscript(node)

        # Continue visiting child nodes
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:
        """
        Visit for loops to detect dynamic column operations.

        Args:
            node: AST For node
        """
        # Try to analyze the loop for column creation patterns
        if self._is_dynamic_column_loop(node):
            # Let the column analyzer try to extract column information
            self.column_analyzer.analyze_for_loop(node)

            # Check if the analysis was successful by seeing if we added column references
            # If not, mark it as dynamic operations requiring manual verification
            if not self._has_analyzable_loop_pattern(node):
                self.dynamic_operations += 1
                self.warnings.append(
                    "Dynamic column operations detected - manual verification recommended",
                )

        # Continue visiting child nodes
        self.generic_visit(node)

    def visit_If(self, node: ast.If) -> None:
        """
        Visit if statements to detect conditional logic complexity.

        Args:
            node: AST If node
        """
        # Check for complex conditional expressions
        if self._is_complex_conditional(node):
            self.complex_expressions += 1

        # Continue visiting child nodes
        self.generic_visit(node)

    def visit_With(self, node: ast.With) -> None:
        """
        Visit with statements (context managers).

        Args:
            node: AST With node
        """
        # Continue visiting child nodes
        self.generic_visit(node)

    def _is_udf_call(self, node: ast.Call) -> bool:
        """
        Check if a function call represents UDF usage.

        Args:
            node: AST Call node

        Returns:
            True if this appears to be a UDF-related call
        """
        # Check for @F.udf() decorator calls
        if isinstance(node.func, ast.Attribute):
            if (
                isinstance(node.func.value, ast.Name)
                and node.func.value.id == "F"
                and node.func.attr == "udf"
            ):
                return True

        # Check for udf() function calls
        if isinstance(node.func, ast.Name) and "udf" in node.func.id.lower():
            return True

        # Check for function names that suggest UDF usage
        if isinstance(node.func, ast.Name):
            func_name = node.func.id.lower()
            udf_indicators = ["_udf", "udf_", "user_defined", "custom_func"]
            if any(indicator in func_name for indicator in udf_indicators):
                return True

        return False

    def _is_dynamic_column_loop(self, node: ast.For) -> bool:
        """
        Check if a for loop appears to create columns dynamically.

        Args:
            node: AST For node

        Returns:
            True if loop likely creates dynamic columns
        """
        # Look for withColumn calls in the loop body
        for child in ast.walk(node):
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute):
                if child.func.attr == "withColumn":
                    return True

        return False

    def _has_analyzable_loop_pattern(self, node: ast.For) -> bool:
        """
        Check if a for loop has a pattern we can analyze statically.

        Args:
            node: AST For node

        Returns:
            True if the loop pattern can be analyzed statically
        """
        # Check if loop iterates over a list (literal or variable) and creates columns with known names
        has_iterable = False

        if isinstance(node.iter, ast.List):
            # Direct list iteration: for item in ["a", "b", "c"]
            all_strings = all(
                isinstance(item, ast.Constant) and isinstance(item.value, str)
                for item in node.iter.elts
            )
            has_iterable = all_strings
        elif isinstance(node.iter, ast.Name):
            # Variable iteration: for item in var_name
            # Assume it's analyzable for now - the column analyzer will try to resolve it
            has_iterable = True

        if has_iterable:
            # Check if loop body contains withColumn with analyzable pattern
            for child in ast.walk(node):
                if (
                    isinstance(child, ast.Call)
                    and isinstance(child.func, ast.Attribute)
                    and child.func.attr == "withColumn"
                ):
                    return True

            # Also check if there are f-string assignments in the loop body
            for stmt in node.body:
                if isinstance(stmt, ast.Assign) and isinstance(
                    stmt.value,
                    ast.JoinedStr,
                ):
                    return True

        return False

    def _is_complex_conditional(self, node: ast.If) -> bool:
        """
        Check if an if statement has complex conditional logic.

        Args:
            node: AST If node

        Returns:
            True if conditional logic is complex
        """
        # Count the depth and complexity of the condition
        condition_complexity = self._calculate_expression_complexity(node.test)

        # Consider it complex if it has multiple boolean operators or deep nesting
        return condition_complexity > 2

    def _calculate_expression_complexity(self, node: ast.AST) -> int:
        """
        Calculate the complexity score of an expression.

        Args:
            node: AST node representing an expression

        Returns:
            Complexity score (higher = more complex)
        """
        if isinstance(node, (ast.BoolOp, ast.Compare)):
            return 1 + sum(
                self._calculate_expression_complexity(child)
                for child in ast.iter_child_nodes(node)
            )
        elif isinstance(node, ast.Call):
            return 1
        else:
            return 0

    def get_analysis_summary(self) -> dict[str, Any]:
        """
        Get comprehensive analysis summary.

        Returns:
            Dictionary with analysis metadata and quality indicators
        """
        return {
            "function_name": self.func_name,
            "function_def_found": self.function_def_found,
            "dataframe_parameters": list(self.df_tracker.get_dataframe_params()),
            "dataframe_variables": list(self.df_tracker.get_all_dataframe_vars()),
            "udf_count": self.udf_count,
            "dynamic_operations": self.dynamic_operations,
            "complex_expressions": self.complex_expressions,
            "errors": self.errors.copy(),
            "warnings": self.warnings.copy(),
        }
