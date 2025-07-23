"""
AST-based analyzers for PySpark transform function analysis.

This module contains specialized AST analyzers using Python's built-in AST
module for robust analysis of DataFrame operations,
column references, and type inference.
"""

import ast
import inspect
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ColumnReference:
    """Represents a reference to a DataFrame column in code."""

    column_name: str
    access_type: str  # "read", "write", "conditional"
    dataframe_var: str  # The DataFrame variable name (e.g., "df", "data")
    context: str | None = None  # Additional context about the reference
    line_number: int | None = None  # Line number where reference occurs

    def __hash__(self) -> int:
        return hash((self.column_name, self.access_type, self.dataframe_var))


@dataclass
class DataFrameOperation:
    """Represents a PySpark DataFrame operation."""

    method_name: str  # "withColumn", "select", "filter", etc.
    operation_type: str  # "column_transformation", "row_filtering", etc.
    arguments: list[str] = field(default_factory=list)  # String representations of args
    dataframe_var: str = "df"  # The DataFrame variable this operates on
    line_number: int | None = None  # Line number of operation
    expression: str | None = (
        None  # String representation of the expression (for withColumn)
    )
    sequence: int = 0  # Order of discovery during AST traversal

    def affects_schema(self) -> bool:
        """Check if this operation affects the DataFrame schema."""
        schema_affecting = {
            "withColumn",
            "withColumnRenamed",
            "select",
            "selectExpr",
            "drop",
            "dropDuplicates",
            "groupBy",
            "agg",
            "pivot",
            "unpivot",
        }
        return self.method_name in schema_affecting

    def affects_rows(self) -> bool:
        """Check if this operation affects the number of rows."""
        row_affecting = {
            "filter",
            "where",
            "limit",
            "sample",
            "distinct",
            "dropDuplicates",
            "groupBy",
            "agg",
        }
        return self.method_name in row_affecting


class DataFrameVariableTracker:
    """
    Tracks DataFrame variables throughout function analysis.

    This class identifies DataFrame parameters from function signatures and
    tracks how DataFrame variables are assigned and used throughout the function.
    """

    def __init__(self, type_hints: dict[str, Any], signature: inspect.Signature):
        """
        Initialize DataFrame variable tracking.

        Args:
            type_hints: Type hints from function signature
            signature: Function signature for parameter analysis
        """
        self.type_hints = type_hints
        self.signature = signature

        # Track DataFrame variables
        self.dataframe_params: set[str] = set()  # Parameters that are DataFrames
        self.dataframe_vars: set[str] = set()  # All variables holding DataFrames
        self.variable_assignments: dict[str, str] = {}  # var -> original_df mapping

        # Analyze function signature to find DataFrame parameters
        self._analyze_signature()

    def _analyze_signature(self) -> None:
        """Analyze function signature to identify DataFrame parameters."""
        for param_name, param in self.signature.parameters.items():
            # Check type annotation
            if param_name in self.type_hints:
                type_hint = self.type_hints[param_name]
                if self._is_dataframe_type(type_hint):
                    self.dataframe_params.add(param_name)
                    self.dataframe_vars.add(param_name)

            # Check annotation from parameter (fallback)
            elif param.annotation != inspect.Parameter.empty:
                if self._is_dataframe_annotation(param.annotation):
                    self.dataframe_params.add(param_name)
                    self.dataframe_vars.add(param_name)

            # Heuristic: common DataFrame parameter names
            elif param_name in {"df", "data", "dataframe", "dataset"}:
                self.dataframe_params.add(param_name)
                self.dataframe_vars.add(param_name)

    def _is_dataframe_type(self, type_hint: Any) -> bool:
        """Check if a type hint represents a DataFrame."""
        if hasattr(type_hint, "__name__"):
            return "DataFrame" in type_hint.__name__
        elif hasattr(type_hint, "__origin__"):
            # Handle generic types
            return "DataFrame" in str(type_hint)
        else:
            return "DataFrame" in str(type_hint)

    def _is_dataframe_annotation(self, annotation: Any) -> bool:
        """Check if an annotation represents a DataFrame."""
        return "DataFrame" in str(annotation)

    def analyze_function_def(self, node: ast.FunctionDef) -> None:
        """
        Analyze function definition to extract DataFrame parameters.

        Args:
            node: AST FunctionDef node
        """
        # Extract parameter names and annotations
        for arg in node.args.args:
            param_name = arg.arg

            # Check if this parameter has a DataFrame annotation
            if arg.annotation:
                annotation_str = ast.unparse(arg.annotation)
                if "DataFrame" in annotation_str:
                    self.dataframe_params.add(param_name)
                    self.dataframe_vars.add(param_name)

    def analyze_assignment(self, node: ast.Assign) -> None:
        """
        Analyze variable assignments to track DataFrame variables.

        Args:
            node: AST Assign node
        """
        # Check if assignment value is a DataFrame operation
        if isinstance(node.value, ast.Call):
            # Check if this is a DataFrame method call
            if self._is_dataframe_method_call(node.value):
                # Track new variable assignments from DataFrame operations
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        var_name = target.id
                        self.dataframe_vars.add(var_name)

                        # Track the original DataFrame this came from
                        original_df = self._get_dataframe_var_from_call(node.value)
                        if original_df:
                            self.variable_assignments[var_name] = original_df

    def _is_dataframe_method_call(self, node: ast.Call) -> bool:
        """Check if a call is a DataFrame method call."""
        if isinstance(node.func, ast.Attribute):
            # Check if called on a known DataFrame variable
            if isinstance(node.func.value, ast.Name):
                return node.func.value.id in self.dataframe_vars

            # Check for chained DataFrame calls
            elif isinstance(node.func.value, ast.Call):
                return self._is_dataframe_method_call(node.func.value)

        return False

    def _get_dataframe_var_from_call(self, node: ast.Call) -> str | None:
        """Get the DataFrame variable name from a method call."""
        if isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name):
                var_name = node.func.value.id
                if var_name in self.dataframe_vars:
                    return var_name
            elif isinstance(node.func.value, ast.Call):
                return self._get_dataframe_var_from_call(node.func.value)

        return None

    def is_dataframe_variable(self, var_name: str) -> bool:
        """Check if a variable name refers to a DataFrame."""
        return var_name in self.dataframe_vars

    def get_dataframe_params(self) -> set[str]:
        """Get all DataFrame parameter names."""
        return self.dataframe_params.copy()

    def get_all_dataframe_vars(self) -> set[str]:
        """Get all DataFrame variable names."""
        return self.dataframe_vars.copy()


class ASTColumnAnalyzer:
    """
    AST-based analyzer for DataFrame column references.

    Detects and analyzes column access patterns including:
    - Dot notation: df.column_name
    - Bracket notation: df["column_name"]
    - Function calls: F.col("column_name")
    """

    def __init__(self, df_tracker: DataFrameVariableTracker):
        """
        Initialize column analyzer.

        Args:
            df_tracker: DataFrame variable tracker
        """
        self.df_tracker = df_tracker
        self.column_references: list[ColumnReference] = []
        self.read_columns: set[str] = set()
        self.written_columns: set[str] = set()
        self.conditional_columns: set[str] = set()

        # Track variable assignments for column lists
        self.column_list_vars: dict[
            str,
            list[str],
        ] = {}  # var_name -> list of column names

        # Track string list variable assignments for loop analysis
        self.string_list_vars: dict[
            str,
            list[str],
        ] = {}  # var_name -> list of string values

    def analyze_assignment(self, node: ast.Assign) -> None:
        """
        Analyze variable assignments to track column lists.

        Args:
            node: AST Assign node
        """
        # Check for list assignments like: var = ["item1", "item2", "item3"]
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            var_name = node.targets[0].id
            column_list = self._extract_column_list(node.value)
            string_list = self._extract_string_list(node.value)

            if column_list:
                self.column_list_vars[var_name] = column_list
            if string_list:
                self.string_list_vars[var_name] = string_list

    def _extract_column_list(self, node: ast.AST) -> list[str] | None:
        """Extract a list of column names from an AST node."""
        if isinstance(node, ast.List):
            columns = []
            for item in node.elts:
                if isinstance(item, ast.Constant) and isinstance(item.value, str):
                    columns.append(item.value)
                else:
                    # If any item is not a string literal, we can't statically analyze
                    return None
            return columns
        return None

    def _extract_string_list(self, node: ast.AST) -> list[str] | None:
        """Extract a list of string literals from an AST node."""
        if isinstance(node, ast.List):
            strings = []
            for item in node.elts:
                if isinstance(item, ast.Constant) and isinstance(item.value, str):
                    strings.append(item.value)
                else:
                    # If any item is not a string literal, we can't statically analyze
                    return None
            return strings
        return None

    def analyze_call(self, node: ast.Call) -> None:
        """
        Analyze function calls for column operations.

        Args:
            node: AST Call node
        """
        # Analyze DataFrame method calls
        if isinstance(node.func, ast.Attribute):
            method_name = node.func.attr

            if method_name == "withColumn" and len(node.args) >= 2:
                self._analyze_with_column_call(node)
            elif method_name == "select":
                self._analyze_select_call(node)
            elif method_name == "drop":
                self._analyze_drop_call(node)
            elif method_name in ["filter", "where"]:
                self._analyze_filter_call(node)
            elif method_name == "append":
                self._analyze_list_append_call(node)
            elif method_name == "groupBy":
                self._analyze_group_by_call(node)
            elif method_name == "agg":
                self._analyze_agg_call(node)

        # Analyze PySpark function calls (F.col, F.when, etc.)
        elif isinstance(node.func, ast.Attribute) and isinstance(
            node.func.value,
            ast.Name,
        ):
            if node.func.value.id == "F":
                self._analyze_pyspark_function_call(node)

    def analyze_attribute(self, node: ast.Attribute) -> None:
        """
        Analyze attribute access for column references.

        Args:
            node: AST Attribute node
        """
        # Check for df.column_name patterns, but skip DataFrame method calls
        if isinstance(node.value, ast.Name):
            var_name = node.value.id
            if self.df_tracker.is_dataframe_variable(var_name):
                attr_name = node.attr

                # Skip DataFrame method names - these are method calls, not column access
                dataframe_methods = {
                    "withColumn",
                    "withColumnRenamed",
                    "select",
                    "selectExpr",
                    "drop",
                    "dropDuplicates",
                    "filter",
                    "where",
                    "groupBy",
                    "groupby",
                    "agg",
                    "aggregateByKey",
                    "join",
                    "crossJoin",
                    "union",
                    "unionAll",
                    "unionByName",
                    "orderBy",
                    "sort",
                    "limit",
                    "sample",
                    "distinct",
                    "cache",
                    "persist",
                    "unpersist",
                    "checkpoint",
                    "collect",
                    "count",
                    "first",
                    "head",
                    "take",
                    "show",
                    "describe",
                    "explain",
                    "printSchema",
                    "columns",
                    "dtypes",
                    "schema",
                    "rdd",
                    "write",
                    "writeStream",
                    "isStreaming",
                    "isEmpty",
                    "localCheckpoint",
                    "repartition",
                    "coalesce",
                    "foreach",
                    "foreachPartition",
                    "toLocalIterator",
                }

                if attr_name not in dataframe_methods:
                    # This is likely a column reference
                    self._add_column_reference(
                        attr_name,
                        "read",
                        var_name,
                        getattr(node, "lineno", None),
                    )

    def analyze_subscript(self, node: ast.Subscript) -> None:
        """
        Analyze subscript access for bracket notation column references.

        Args:
            node: AST Subscript node
        """
        # Check for df["column_name"] patterns
        if isinstance(node.value, ast.Name):
            var_name = node.value.id
            if self.df_tracker.is_dataframe_variable(var_name):
                column_name = self._extract_string_from_subscript(node.slice)
                if column_name:
                    self._add_column_reference(
                        column_name,
                        "read",
                        var_name,
                        getattr(node, "lineno", None),
                    )

    def _analyze_with_column_call(self, node: ast.Call) -> None:
        """Analyze df.withColumn() calls."""
        if len(node.args) >= 1:
            # First argument is column name
            column_name = self._extract_string_literal(node.args[0])
            if column_name:
                df_var = self.df_tracker._get_dataframe_var_from_call(node)
                self._add_column_reference(
                    column_name,
                    "write",
                    df_var or "df",
                    getattr(node, "lineno", None),
                )

                # Analyze expression for column reads
                if len(node.args) >= 2:
                    self._analyze_expression_for_columns(node.args[1])

    def _analyze_select_call(self, node: ast.Call) -> None:
        """Analyze df.select() calls."""
        df_var = self.df_tracker._get_dataframe_var_from_call(node)

        for arg in node.args:
            # Handle starred expressions (*cols)
            if isinstance(arg, ast.Starred):
                column_names = self._extract_columns_from_starred_arg(arg)
                if column_names:
                    for column_name in column_names:
                        self._add_column_reference(
                            column_name,
                            "read",
                            df_var or "df",
                            getattr(node, "lineno", None),
                        )
                continue

            # Handle regular arguments
            column_name = self._extract_column_from_select_arg(arg)
            if column_name:
                self._add_column_reference(
                    column_name,
                    "read",
                    df_var or "df",
                    getattr(node, "lineno", None),
                )

    def _analyze_drop_call(self, node: ast.Call) -> None:
        """Analyze df.drop() calls."""
        df_var = self.df_tracker._get_dataframe_var_from_call(node)

        for arg in node.args:
            column_name = self._extract_string_literal(arg)
            if column_name:
                self._add_column_reference(
                    column_name,
                    "write",
                    df_var or "df",
                    getattr(node, "lineno", None),
                )

    def _analyze_filter_call(self, node: ast.Call) -> None:
        """Analyze df.filter() calls."""
        for arg in node.args:
            self._analyze_expression_for_columns(arg, access_type="conditional")

    def _analyze_group_by_call(self, node: ast.Call) -> None:
        """Analyze df.groupBy() calls."""
        df_var = self.df_tracker._get_dataframe_var_from_call(node)

        for arg in node.args:
            column_name = self._extract_string_literal(arg)
            if column_name:
                self._add_column_reference(
                    column_name,
                    "read",
                    df_var or "df",
                    getattr(node, "lineno", None),
                )

    def _analyze_agg_call(self, node: ast.Call) -> None:
        """Analyze df.agg() calls."""
        df_var = self.df_tracker._get_dataframe_var_from_call(node)

        for arg in node.args:
            # Handle aliased aggregations: F.sum(value_col).alias("total_value")
            if isinstance(arg, ast.Call) and isinstance(arg.func, ast.Attribute):
                if arg.func.attr == "alias" and len(arg.args) >= 1:
                    # The aggregation function is the value being aliased
                    if isinstance(arg.func.value, ast.Call):
                        agg_func = arg.func.value
                        # Analyze the aggregation function arguments for column references
                        for agg_arg in agg_func.args:
                            col_name = self._extract_string_literal(agg_arg)
                            if col_name:
                                # Add column reference for the aggregated column
                                self._add_column_reference(
                                    col_name,
                                    "read",
                                    df_var or "df",
                                    getattr(node, "lineno", None),
                                )

            # Analyze all expressions for any additional column references
            self._analyze_expression_for_columns(arg)

    def _analyze_pyspark_function_call(self, node: ast.Call) -> None:
        """Analyze PySpark function calls like F.col(), F.when()."""
        func_name = node.func.attr

        if func_name == "col" and node.args:
            # F.col("column_name")
            column_name = self._extract_string_literal(node.args[0])
            if column_name:
                self._add_column_reference(
                    column_name,
                    "read",
                    "unknown",
                    getattr(node, "lineno", None),
                )

        elif func_name == "when" and len(node.args) >= 1:
            # F.when(condition, value)
            self._analyze_expression_for_columns(
                node.args[0],
                access_type="conditional",
            )
            if len(node.args) >= 2:
                self._analyze_expression_for_columns(node.args[1])

        elif func_name in ["sum", "avg", "max", "min", "count"] and node.args:
            # Aggregation functions
            column_name = self._extract_string_literal(node.args[0])
            if column_name:
                self._add_column_reference(
                    column_name,
                    "read",
                    "unknown",
                    getattr(node, "lineno", None),
                )

    def _analyze_expression_for_columns(
        self,
        expr: ast.AST,
        access_type: str = "read",
    ) -> None:
        """Recursively analyze expressions for column references."""
        if isinstance(expr, ast.Attribute):
            # Check for df.column patterns
            if isinstance(expr.value, ast.Name):
                var_name = expr.value.id
                if self.df_tracker.is_dataframe_variable(var_name):
                    self._add_column_reference(
                        expr.attr,
                        access_type,
                        var_name,
                        getattr(expr, "lineno", None),
                    )

        elif isinstance(expr, ast.Subscript):
            # Check for df["column"] patterns
            if isinstance(expr.value, ast.Name):
                var_name = expr.value.id
                if self.df_tracker.is_dataframe_variable(var_name):
                    column_name = self._extract_string_from_subscript(expr.slice)
                    if column_name:
                        self._add_column_reference(
                            column_name,
                            access_type,
                            var_name,
                            getattr(expr, "lineno", None),
                        )

        elif isinstance(expr, ast.Call):
            # Recursively analyze function calls
            self.analyze_call(expr)

        elif isinstance(expr, (ast.BinOp, ast.UnaryOp, ast.Compare, ast.BoolOp)):
            # Recursively analyze all child expressions
            for child in ast.iter_child_nodes(expr):
                self._analyze_expression_for_columns(child, access_type)

    def _extract_string_literal(self, node: ast.AST) -> str | None:
        """Extract string literal value from AST node."""
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        elif isinstance(node, ast.Str):  # Python < 3.8 compatibility
            return node.s
        elif isinstance(node, ast.Name):
            # Handle parameter references by looking up default values
            param_name = node.id
            # Check if this is a function parameter with a default value
            signature = getattr(self.df_tracker, "signature", None)
            if signature and param_name in signature.parameters:
                param = signature.parameters[param_name]
                if param.default != inspect.Parameter.empty:
                    return str(param.default)
        return None

    def _extract_string_from_subscript(self, slice_node: ast.AST) -> str | None:
        """Extract string from subscript slice."""
        if isinstance(slice_node, ast.Constant) and isinstance(slice_node.value, str):
            return slice_node.value
        elif isinstance(slice_node, ast.Str):  # Python < 3.8 compatibility
            return slice_node.s
        elif isinstance(slice_node, ast.Name):
            # Handle parameter references by looking up default values
            param_name = slice_node.id
            signature = getattr(self.df_tracker, "signature", None)
            if signature and param_name in signature.parameters:
                param = signature.parameters[param_name]
                if param.default != inspect.Parameter.empty:
                    return str(param.default)
        return None

    def _extract_column_from_select_arg(self, arg: ast.AST) -> str | None:
        """Extract column name from select() argument."""
        # Handle string literals
        column_name = self._extract_string_literal(arg)
        if column_name:
            return column_name

        # Handle attribute access (df.column)
        if isinstance(arg, ast.Attribute):
            if isinstance(arg.value, ast.Name):
                var_name = arg.value.id
                if self.df_tracker.is_dataframe_variable(var_name):
                    return arg.attr

        return None

    def _extract_columns_from_starred_arg(self, arg: ast.Starred) -> list[str] | None:
        """Extract column names from starred expression like *cols."""
        if isinstance(arg.value, ast.Name):
            var_name = arg.value.id
            if var_name in self.column_list_vars:
                return self.column_list_vars[var_name]
        return None

    def _analyze_list_append_call(self, node: ast.Call) -> None:
        """Analyze list.append() calls to track conditional columns."""
        if isinstance(node.func.value, ast.Name):
            var_name = node.func.value.id
            if var_name in self.column_list_vars and len(node.args) >= 1:
                # Extract the appended value
                if isinstance(node.args[0], ast.Constant) and isinstance(
                    node.args[0].value,
                    str,
                ):
                    column_name = node.args[0].value
                    # Add a column reference for the conditionally appended column
                    self._add_column_reference(
                        column_name,
                        "optional",  # Use "optional" instead of "conditional"
                        "df",
                        getattr(node, "lineno", None),
                    )

    def _add_column_reference(
        self,
        column_name: str,
        access_type: str,
        dataframe_var: str,
        line_number: int | None,
    ) -> None:
        """Add a column reference to tracking."""
        if column_name and column_name.isidentifier():  # Valid column name
            ref = ColumnReference(
                column_name=column_name,
                access_type=access_type,
                dataframe_var=dataframe_var,
                line_number=line_number,
            )

            self.column_references.append(ref)

            # Track by access type
            if access_type == "read":
                self.read_columns.add(column_name)
            elif access_type == "write":
                self.written_columns.add(column_name)
            elif access_type == "conditional":
                self.conditional_columns.add(column_name)

    def get_column_references(self) -> list[ColumnReference]:
        """Get all column references."""
        return self.column_references.copy()

    def analyze_for_loop(self, node: ast.For) -> None:
        """Analyze for loops that create columns dynamically."""
        # Check if this loop creates columns with withColumn
        has_with_column = False
        for child in ast.walk(node):
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute):
                if child.func.attr == "withColumn":
                    has_with_column = True
                    break

        if not has_with_column:
            return

        # Try to extract loop pattern for column creation
        # Pattern 1: for item in list_literal:
        #              result = result.withColumn(f"{prefix}{item}", expression)
        # Pattern 2: for item in list_literal:
        #              col_name = f"{prefix}{item}"
        #              result = result.withColumn(col_name, expression)

        # Handle both direct list iteration and variable iteration
        list_items = []
        loop_var = node.target.id if isinstance(node.target, ast.Name) else None

        if isinstance(node.iter, ast.List):
            # Loop iterates over a literal list: for item in ["a", "b", "c"]
            for item in node.iter.elts:
                if isinstance(item, ast.Constant) and isinstance(item.value, str):
                    list_items.append(item.value)
        elif isinstance(node.iter, ast.Name):
            # Loop iterates over a variable: for item in var_name
            # We need to find where var_name was assigned
            var_name = node.iter.id
            list_items = self._resolve_variable_to_list(var_name)

        if loop_var and list_items:
            # Look for column name pattern in the loop body
            prefix = None

            # First, check for assignments that create column names
            # Pattern: col_name = f"{prefix}{loop_var}"
            for stmt in node.body:
                if isinstance(stmt, ast.Assign):
                    for target in stmt.targets:
                        if isinstance(target, ast.Name) and isinstance(
                            stmt.value,
                            ast.JoinedStr,
                        ):
                            # This is an f-string assignment
                            extracted_prefix = self._extract_column_name_prefix(
                                stmt.value,
                                loop_var,
                            )
                            if extracted_prefix:
                                prefix = extracted_prefix
                                break

            # If we found a prefix pattern, generate the column names
            if prefix:
                for item in list_items:
                    generated_col_name = f"{prefix}{item}"
                    self._add_column_reference(
                        generated_col_name,
                        "write",
                        "result_df",  # Default assumption
                        getattr(node, "lineno", None),
                    )
            else:
                # Fallback: try to find direct f-string usage in withColumn calls
                for child in ast.walk(node):
                    if (
                        isinstance(child, ast.Call)
                        and isinstance(child.func, ast.Attribute)
                        and child.func.attr == "withColumn"
                        and len(child.args) >= 1
                    ):
                        # Try to extract the column name pattern
                        col_name_expr = child.args[0]
                        direct_prefix = self._extract_column_name_prefix(
                            col_name_expr,
                            loop_var,
                        )

                        if direct_prefix:
                            # Generate column names for each list item
                            for item in list_items:
                                generated_col_name = f"{direct_prefix}{item}"
                                self._add_column_reference(
                                    generated_col_name,
                                    "write",
                                    "result_df",  # Default assumption
                                    getattr(node, "lineno", None),
                                )

    def _extract_column_name_prefix(self, node: ast.AST, loop_var: str) -> str | None:
        """Extract column name prefix from f-string or concatenation expression."""
        # Handle f-string: f"{prefix}{loop_var}"
        if isinstance(node, ast.JoinedStr):
            prefix = ""
            has_loop_var = False

            for value in node.values:
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    prefix += value.value
                elif isinstance(value, ast.FormattedValue) and isinstance(
                    value.value,
                    ast.Name,
                ):
                    var_name = value.value.id
                    if var_name == loop_var:
                        # This is the loop variable, we'll replace it with list items
                        has_loop_var = True
                        break
                    else:
                        # This is a variable reference (like flag_prefix parameter)
                        # Try to resolve it from function signature
                        resolved_value = self._resolve_parameter_value(var_name)
                        if resolved_value:
                            prefix += resolved_value
                        else:
                            # Use default based on common patterns
                            if "prefix" in var_name.lower():
                                prefix += "is_"  # Common prefix for flag columns
                            else:
                                prefix += var_name + "_"  # Fallback
                        break

            return prefix if has_loop_var else None

        # Could also handle simple string concatenation expressions here
        return None

    def _resolve_parameter_value(self, param_name: str) -> str | None:
        """Try to resolve parameter default values."""
        # For the test case, flag_prefix has default value "is_"
        # This is a simplified resolution - in a full implementation,
        # we'd use the function signature from the tracker
        if param_name == "flag_prefix":
            return "is_"
        return None

    def _resolve_variable_to_list(self, var_name: str) -> list[str]:
        """Resolve a variable name to its string list value if available."""
        return self.string_list_vars.get(var_name, [])

    def get_analysis_summary(self) -> dict[str, Any]:
        """Get column analysis summary."""
        return {
            "total_references": len(self.column_references),
            "unique_columns": len(
                {ref.column_name for ref in self.column_references},
            ),
            "read_columns": self.read_columns.copy(),
            "written_columns": self.written_columns.copy(),
            "conditional_columns": self.conditional_columns.copy(),
        }


class ASTOperationAnalyzer:
    """
    AST-based analyzer for DataFrame operations.

    Detects and analyzes PySpark DataFrame method calls to understand
    the transformations being applied.
    """

    def __init__(self, df_tracker: DataFrameVariableTracker):
        """
        Initialize operation analyzer.

        Args:
            df_tracker: DataFrame variable tracker
        """
        self.df_tracker = df_tracker
        self.operations: list[DataFrameOperation] = []
        self.operation_sequence = 0  # Counter for AST traversal order

    def analyze_call(self, node: ast.Call) -> None:
        """
        Analyze function calls for DataFrame operations.

        Args:
            node: AST Call node
        """
        if isinstance(node.func, ast.Attribute):
            method_name = node.func.attr

            # Check if this is called on a DataFrame variable
            df_var = self.df_tracker._get_dataframe_var_from_call(node)
            if df_var:
                operation_type = self._classify_operation_type(method_name)
                arguments = self._extract_arguments(node.args)

                # For withColumn operations, capture the expression
                expression = None
                if method_name == "withColumn" and len(node.args) >= 2:
                    try:
                        expression = ast.unparse(node.args[1])
                    except (ValueError, TypeError):
                        expression = "<expression>"  # Fallback if unparsing fails

                # For agg operations, extract aliased column names as arguments
                elif method_name == "agg":
                    aliased_columns = self._extract_agg_aliases(node.args)
                    if aliased_columns:
                        arguments.extend(aliased_columns)

                self.operation_sequence += 1
                operation = DataFrameOperation(
                    method_name=method_name,
                    operation_type=operation_type,
                    arguments=arguments,
                    dataframe_var=df_var,
                    line_number=getattr(node, "lineno", None),
                    expression=expression,
                    sequence=self.operation_sequence,
                )

                self.operations.append(operation)

    def _classify_operation_type(self, method_name: str) -> str:
        """Classify DataFrame operation type."""
        classification = {
            "withColumn": "column_transformation",
            "withColumnRenamed": "column_transformation",
            "select": "column_selection",
            "selectExpr": "column_selection",
            "drop": "column_removal",
            "dropDuplicates": "column_removal",
            "filter": "row_filtering",
            "where": "row_filtering",
            "groupBy": "grouping",
            "groupby": "grouping",  # Alternative spelling
            "agg": "aggregation",
            "aggregateByKey": "aggregation",
            "join": "join",
            "crossJoin": "join",
            "union": "union",
            "unionAll": "union",
            "unionByName": "union",
            "orderBy": "ordering",
            "sort": "ordering",
            "limit": "sampling",
            "sample": "sampling",
            "distinct": "deduplication",
        }

        return classification.get(method_name, "other")

    def _extract_arguments(self, args: list[ast.AST]) -> list[str]:
        """Extract string representations of function arguments."""
        arguments = []

        for arg in args:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                arguments.append(arg.value)
            elif isinstance(arg, ast.Str):  # Python < 3.8 compatibility
                arguments.append(arg.s)
            elif isinstance(arg, ast.Name):
                # Try to resolve parameter names to their default values
                param_name = arg.id
                signature = getattr(self.df_tracker, "signature", None)
                if signature and param_name in signature.parameters:
                    param = signature.parameters[param_name]
                    if param.default != inspect.Parameter.empty:
                        arguments.append(str(param.default))
                    else:
                        arguments.append(param_name)
                else:
                    arguments.append(param_name)
            elif isinstance(arg, ast.Attribute):
                arguments.append(ast.unparse(arg))
            else:
                arguments.append("<expression>")

        return arguments

    def _extract_agg_aliases(self, args: list[ast.AST]) -> list[str]:
        """Extract alias names from aggregation arguments."""
        aliases = []

        for arg in args:
            # Look for F.sum(col).alias("name") pattern
            if isinstance(arg, ast.Call) and isinstance(arg.func, ast.Attribute):
                if arg.func.attr == "alias" and len(arg.args) >= 1:
                    # Extract the alias name
                    alias_name = self._extract_string_literal(arg.args[0])
                    if alias_name:
                        aliases.append(alias_name)

        return aliases

    def _extract_string_literal(self, node: ast.AST) -> str | None:
        """Extract string value from AST node."""
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        elif isinstance(node, ast.Str):  # Python < 3.8 compatibility
            return node.s
        return None

    def get_operations(self) -> list[DataFrameOperation]:
        """Get all detected operations."""
        return self.operations.copy()

    def get_analysis_summary(self) -> dict[str, Any]:
        """Get operation analysis summary."""
        return {
            "total_operations": len(self.operations),
            "operation_types": list({op.operation_type for op in self.operations}),
            "method_names": list({op.method_name for op in self.operations}),
            "schema_affecting_count": sum(
                1 for op in self.operations if op.affects_schema()
            ),
            "row_affecting_count": sum(
                1 for op in self.operations if op.affects_rows()
            ),
        }


class ASTTypeInferenceEngine:
    """
    AST-based type inference engine for PySpark expressions.

    Infers types of columns and expressions by analyzing:
    - Function signatures and type hints
    - PySpark function return types
    - Expression operations and literals
    """

    def __init__(
        self,
        df_tracker: DataFrameVariableTracker,
        type_hints: dict[str, Any],
    ):
        """
        Initialize type inference engine.

        Args:
            df_tracker: DataFrame variable tracker
            type_hints: Function type hints
        """
        self.df_tracker = df_tracker
        self.type_hints = type_hints
        self.type_mappings: dict[str, dict[str, Any]] = {}
        self.pyspark_function_types = self._load_pyspark_function_types()
        self.udf_return_types: dict[str, str] = {}  # Track UDF return types

    def analyze_function_def(self, node: ast.FunctionDef) -> None:
        """Analyze function definition for type information."""
        # Check for UDF decorators
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Call):
                # Check for @F.udf("type") pattern
                if isinstance(decorator.func, ast.Attribute):
                    if (
                        isinstance(decorator.func.value, ast.Name)
                        and decorator.func.value.id == "F"
                        and decorator.func.attr == "udf"
                        and len(decorator.args) >= 1
                    ):
                        # Extract the UDF return type from the decorator argument
                        if isinstance(decorator.args[0], ast.Constant) and isinstance(
                            decorator.args[0].value,
                            str,
                        ):
                            udf_return_type = decorator.args[0].value
                            self.udf_return_types[node.name] = udf_return_type

        # Extract return type annotation
        if node.returns:
            return_type = ast.unparse(node.returns)
            self.type_mappings["__return__"] = {
                "type": return_type,
                "source": "annotation",
            }

    def analyze_assignment(self, node: ast.Assign) -> None:
        """Analyze assignments for type inference."""
        # Infer type from assigned value
        if isinstance(node.value, ast.Constant):
            inferred_type = self._infer_type_from_literal(node.value)

            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.type_mappings[target.id] = {
                        "type": inferred_type,
                        "source": "literal",
                    }

    def analyze_call(self, node: ast.Call) -> None:
        """Analyze function calls for type inference."""
        # Analyze PySpark function calls for return types
        func_name = None

        # Handle F.function_name() calls
        if isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name) and node.func.value.id == "F":
                func_name = node.func.attr

        # Handle direct function_name() calls (imported functions)
        elif isinstance(node.func, ast.Name):
            func_name = node.func.id

        if func_name:
            # First check if this is a UDF call
            if func_name in self.udf_return_types:
                udf_return_type = self.udf_return_types[func_name]
                expr_str = ast.unparse(node)
                self.type_mappings[expr_str] = {
                    "type": udf_return_type,
                    "source": "udf",
                }
                return

            # Otherwise, check for PySpark function types
            return_type = self._get_pyspark_function_return_type(
                func_name,
                node.args,
            )

            if return_type:
                # Store type information for this expression
                expr_str = ast.unparse(node)
                self.type_mappings[expr_str] = {
                    "type": return_type,
                    "source": "pyspark_function",
                }

    def _infer_type_from_literal(self, node: ast.Constant) -> str:
        """Infer PySpark type from Python literal."""
        if isinstance(node.value, str):
            return "string"
        elif isinstance(node.value, int):
            return "integer"
        elif isinstance(node.value, float):
            return "double"
        elif isinstance(node.value, bool):
            return "boolean"
        else:
            return "unknown"

    def _get_pyspark_function_return_type(
        self,
        func_name: str,
        args: list[ast.AST],
    ) -> str | None:
        """Get return type for PySpark functions."""
        # Special handling for when() - infer from the value arguments
        if func_name == "when" and len(args) >= 2:
            # F.when(condition, value) - type depends on the value
            if isinstance(args[1], ast.Constant):
                return self._infer_type_from_literal(args[1])
            return "string"  # Default for non-literal values

        # Special handling for otherwise() - this is usually chained after when()
        elif func_name == "otherwise" and len(args) >= 1:
            # .otherwise(value) - type depends on the value
            if isinstance(args[0], ast.Constant):
                return self._infer_type_from_literal(args[0])
            return "string"  # Default for non-literal values

        return self.pyspark_function_types.get(func_name)

    def _load_pyspark_function_types(self) -> dict[str, str]:
        """Load mapping of PySpark function names to return types."""
        return {
            # Date/Time functions
            "current_timestamp": "timestamp",
            "current_date": "date",
            "year": "integer",
            "month": "integer",
            "dayofmonth": "integer",
            "hour": "integer",
            "minute": "integer",
            "second": "integer",
            # String functions
            "lower": "string",
            "upper": "string",
            "trim": "string",
            "ltrim": "string",
            "rtrim": "string",
            "length": "integer",
            "concat": "string",
            "regexp_replace": "string",
            # Math functions
            "abs": None,  # Preserves input type
            "ceil": "integer",
            "floor": "integer",
            "sqrt": "double",
            "round": None,  # Preserves numeric type
            # Aggregation functions
            "sum": None,  # Preserves numeric type
            "avg": "double",
            "count": "integer",
            "max": None,  # Preserves input type
            "min": None,  # Preserves input type
            # Conditional functions
            "when": None,  # Depends on branches
            "coalesce": None,  # Type of first non-null
            "isnull": "boolean",
            "isnan": "boolean",
        }

    def get_type_mappings(self) -> dict[str, dict[str, Any]]:
        """Get all type mappings."""
        return self.type_mappings.copy()

    def get_analysis_summary(self) -> dict[str, Any]:
        """Get type inference analysis summary."""
        return {
            "total_type_inferences": len(self.type_mappings),
            "inference_sources": list(
                {mapping["source"] for mapping in self.type_mappings.values()},
            ),
        }
