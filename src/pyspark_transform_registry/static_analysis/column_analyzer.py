"""
Column reference analyzer for PySpark transform functions.

This module analyzes DataFrame column references in PySpark code to identify
which columns are being accessed, added, modified, or removed.
"""

from typing import Any

import libcst as cst


class ColumnReference:
    """Represents a reference to a DataFrame column in code."""

    def __init__(
        self,
        column_name: str,
        access_type: str,
    ):
        self.column_name = column_name
        self.access_type = access_type  # "read", "write", "conditional"
        self.context = None  # Additional context about the reference

    def __eq__(self, other):
        return (
            self.column_name == other.column_name
            and self.access_type == other.access_type
        )

    def __hash__(self):
        return hash((self.column_name, self.access_type))

    def __repr__(self):
        return f"ColumnReference(column_name={self.column_name}, access_type={self.access_type})"


class ColumnAnalyzer(cst.CSTVisitor):
    """
    Analyzes column references in PySpark transform function code.

    This analyzer identifies:
    - Column reads: df.column_name, df["column_name"], F.col("column_name")
    - Column writes: df.withColumn(), df.select(), df.drop()
    - Conditional access: df.filter(), F.when() conditions
    """

    def __init__(self):
        self.column_references: set[ColumnReference] = set()
        self.detected_columns: set[str] = set()
        self.operation_contexts: list[str] = []

        # Track DataFrame variable names
        self.dataframe_vars: set[str] = {"df"}  # Common convention

        # Track column operations
        self.read_columns: set[str] = set()
        self.written_columns: set[str] = set()
        self.conditional_columns: set[str] = set()

    def __eq__(self, other):
        return (
            self.column_references == other.column_references
            and self.detected_columns == other.detected_columns
            and self.operation_contexts == other.operation_contexts
            and self.dataframe_vars == other.dataframe_vars
            and self.read_columns == other.read_columns
            and self.written_columns == other.written_columns
            and self.conditional_columns == other.conditional_columns
        )

    def visit_Call(self, node: cst.Call) -> None:
        """Visit function calls to detect column operations."""
        # Analyze different types of column operations
        self._analyze_withColumn_call(node)
        self._analyze_select_call(node)
        self._analyze_drop_call(node)
        self._analyze_filter_call(node)
        self._analyze_functions_call(node)
        self._analyze_groupBy_call(node)
        self._analyze_agg_call(node)

    def visit_Attribute(self, node: cst.Attribute) -> None:
        """Visit attribute access to detect df.column_name patterns."""
        # Check if this is a DataFrame column access like df.column_name
        if isinstance(node.value, cst.Name):
            var_name = node.value.value
            if var_name in self.dataframe_vars:
                column_name = node.attr.value
                self._add_column_reference(column_name, "read")

    def visit_Subscript(self, node: cst.Subscript) -> None:
        """Visit subscript access to detect df["column_name"] patterns."""
        # Check if this is DataFrame bracket notation like df["column"]
        if isinstance(node.value, cst.Name):
            var_name = node.value.value
            if var_name in self.dataframe_vars:
                # Extract column name from subscript
                column_name = self._extract_string_from_subscript(node.slice[0].slice)
                if column_name:
                    self._add_column_reference(column_name, "read")

    def _analyze_withColumn_call(self, node: cst.Call) -> None:
        """Analyze df.withColumn() calls."""
        if self._is_method_call(node, "withColumn"):
            args = node.args
            if len(args) >= 1:
                # First argument is the column name
                column_name = self._extract_string_from_arg(args[0])
                if column_name:
                    self._add_column_reference(column_name, "write")

                # Second argument may reference other columns
                if len(args) >= 2:
                    self._analyze_expression_for_columns(args[1].value)

    def _analyze_select_call(self, node: cst.Call) -> None:
        """Analyze df.select() calls."""
        if self._is_method_call(node, "select"):
            for arg in node.args:
                column_name = self._extract_column_from_select_arg(arg)
                if column_name:
                    self._add_column_reference(column_name, "read")

    def _analyze_drop_call(self, node: cst.Call) -> None:
        """Analyze df.drop() calls."""
        if self._is_method_call(node, "drop"):
            for arg in node.args:
                column_name = self._extract_string_from_arg(arg)
                if column_name:
                    self._add_column_reference(
                        column_name,
                        "write",
                    )  # Removing is a write operation

    def _analyze_filter_call(self, node: cst.Call) -> None:
        """Analyze df.filter() calls."""
        if self._is_method_call(node, "filter"):
            for arg in node.args:
                self._analyze_expression_for_columns(
                    arg.value,
                    access_type="conditional",
                )

    def _analyze_functions_call(self, node: cst.Call) -> None:
        """Analyze PySpark functions like F.col(), F.when(), etc."""
        if isinstance(node.func, cst.Attribute):
            if isinstance(node.func.value, cst.Name) and node.func.value.value == "F":
                func_name = node.func.attr.value

                if func_name == "col" and node.args:
                    # F.col("column_name")
                    column_name = self._extract_string_from_arg(node.args[0])
                    if column_name:
                        self._add_column_reference(column_name, "read")

                elif func_name == "when" and len(node.args) >= 1:
                    # F.when(condition, value) - analyze condition for columns
                    self._analyze_expression_for_columns(
                        node.args[0].value,
                        access_type="conditional",
                    )

                elif func_name in ["sum", "avg", "max", "min", "count"] and node.args:
                    # Aggregation functions
                    column_name = self._extract_string_from_arg(node.args[0])
                    if column_name:
                        self._add_column_reference(column_name, "read")

    def _analyze_groupBy_call(self, node: cst.Call) -> None:
        """Analyze df.groupBy() calls."""
        if self._is_method_call(node, "groupBy"):
            for arg in node.args:
                column_name = self._extract_string_from_arg(arg)
                if column_name:
                    self._add_column_reference(column_name, "read")

    def _analyze_agg_call(self, node: cst.Call) -> None:
        """Analyze df.agg() calls."""
        if self._is_method_call(node, "agg"):
            for arg in node.args:
                self._analyze_expression_for_columns(arg.value)

    def _analyze_expression_for_columns(
        self,
        expr: cst.BaseExpression,
        access_type: str = "read",
    ) -> None:
        """Recursively analyze an expression for column references."""
        if isinstance(expr, cst.Attribute):
            # Check for df.column patterns
            if (
                isinstance(expr.value, cst.Name)
                and expr.value.value in self.dataframe_vars
            ):
                self._add_column_reference(expr.attr.value, access_type)

        elif isinstance(expr, cst.Subscript):
            # Check for df["column"] patterns
            if (
                isinstance(expr.value, cst.Name)
                and expr.value.value in self.dataframe_vars
            ):
                column_name = self._extract_string_from_subscript(expr.slice)
                if column_name:
                    self._add_column_reference(column_name, access_type)

        elif isinstance(expr, cst.Call):
            # Recursively analyze function calls
            if isinstance(expr.func, cst.Attribute):
                if (
                    isinstance(expr.func.value, cst.Name)
                    and expr.func.value.value == "F"
                    and expr.func.attr.value == "col"
                    and expr.args
                ):
                    column_name = self._extract_string_from_arg(expr.args[0])
                    if column_name:
                        self._add_column_reference(column_name, access_type)

            # Analyze arguments
            for arg in expr.args:
                self._analyze_expression_for_columns(arg.value, access_type)

        elif isinstance(expr, cst.BinaryOperation):
            # Analyze both sides of binary operations
            self._analyze_expression_for_columns(expr.left, access_type)
            self._analyze_expression_for_columns(expr.right, access_type)

        elif isinstance(expr, cst.UnaryOperation):
            # Analyze unary operations
            self._analyze_expression_for_columns(expr.expression, access_type)

    def _extract_column_from_select_arg(self, arg: cst.Arg) -> str | None:
        """Extract column name from select() argument."""
        # Handle string literals
        column_name = self._extract_string_from_arg(arg)
        if column_name:
            return column_name

        # Handle attribute access (df.column)
        if isinstance(arg.value, cst.Attribute):
            if (
                isinstance(arg.value.value, cst.Name)
                and arg.value.value.value in self.dataframe_vars
            ):
                return arg.value.attr.value

        return None

    def _is_method_call(self, node: cst.Call, method_name: str) -> bool:
        """Check if a call is a method call with specific name."""
        if isinstance(node.func, cst.Attribute):
            return node.func.attr.value == method_name
        return False

    def _extract_string_from_arg(self, arg: cst.Arg) -> str | None:
        """Extract string value from function argument."""
        if isinstance(arg.value, cst.SimpleString):
            # Remove quotes and return string content
            return arg.value.value.strip("'\"")
        return None

    def _extract_string_from_subscript(self, slice_expr) -> str | None:
        """Extract string from subscript slice."""
        if isinstance(slice_expr, cst.Index):
            if isinstance(slice_expr.value, cst.SimpleString):
                return slice_expr.value.value.strip("'\"")
        return None

    def _add_column_reference(self, column_name: str, access_type: str) -> None:
        """Add a column reference to our tracking."""
        if column_name and column_name.isidentifier():  # Valid column name
            self.detected_columns.add(column_name)
            self.column_references.add(ColumnReference(column_name, access_type))

            # Track by access type
            if access_type == "read":
                self.read_columns.add(column_name)
            elif access_type == "write":
                self.written_columns.add(column_name)
            elif access_type == "conditional":
                self.conditional_columns.add(column_name)

    def get_analysis_summary(self) -> dict[str, Any]:
        """Get summary of column analysis."""
        return {
            "total_references": len(self.column_references),
            "unique_columns": len(self.detected_columns),
            "read_columns": set(self.read_columns),
            "written_columns": set(self.written_columns),
            "conditional_columns": set(self.conditional_columns),
            "all_detected": set(self.detected_columns),
        }


def find_column_references(source_code: str) -> dict[str, Any]:
    """
    Find all column references in PySpark source code.

    Args:
        source_code: Python source code to analyze

    Returns:
        Dictionary with analysis results
    """
    tree = cst.parse_module(source_code)
    analyzer = ColumnAnalyzer()
    tree.visit(analyzer)
    return analyzer.get_analysis_summary()
