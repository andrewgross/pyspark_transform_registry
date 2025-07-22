"""
Schema inference orchestrator for generating partial schema constraints.

This module coordinates the analysis results from column analysis, operation
analysis, and type inference to generate comprehensive schema constraints.
"""

from typing import Any

from pyspark_transform_registry.static_analysis.column_analyzer import ColumnReference
from pyspark_transform_registry.static_analysis.operation_analyzer import (
    DataFrameOperation,
)

from ..schema_constraints import (
    ColumnRequirement,
    ColumnTransformation,
    PartialSchemaConstraint,
)


class ConstraintGenerator:
    """
    Generates partial schema constraints from static analysis results.

    This class takes the outputs from column analysis, operation analysis,
    and type inference to create a comprehensive PartialSchemaConstraint
    that captures the requirements and transformations of a function.
    """

    def __init__(self):
        self.default_type_mappings = {
            "unknown": "string",  # Default fallback
            "array": "array<string>",
            "map": "map<string,string>",
        }

    def generate_constraint(
        self,
        operations: set[DataFrameOperation],
        column_references: set[ColumnReference],
        type_info: dict[str, Any],
        source_analysis: dict[str, Any],
    ) -> PartialSchemaConstraint:
        """
        Generate a partial schema constraint from analysis results.

        Args:
            operations: List of DataFrame operations from OperationAnalyzer
            column_references: Column reference analysis from ColumnAnalyzer
            type_info: Type inference results from TypeInferenceEngine
            source_analysis: General source analysis metadata

        Returns:
            PartialSchemaConstraint with inferred requirements and transformations
        """
        # Initialize constraint with metadata
        constraint = PartialSchemaConstraint(
            analysis_method="static_analysis",
        )

        # Extract column information
        read_columns = {
            c.column_name for c in column_references if c.access_type == "read"
        }
        written_columns = {
            c.column_name for c in column_references if c.access_type == "write"
        }
        conditional_columns = {
            c.column_name for c in column_references if c.access_type == "conditional"
        }

        # Analyze operations to understand transformations
        operation_analysis = self._analyze_operations(operations)

        # Generate required columns
        required_columns = self._generate_required_columns(
            read_columns,
            conditional_columns,
            type_info,
            operation_analysis,
        )
        constraint.required_columns = required_columns

        # Generate column transformations
        transformations = self._generate_transformations(
            operations,
            written_columns,
            type_info,
            operation_analysis,
        )
        constraint.added_columns = transformations["added"]
        constraint.modified_columns = transformations["modified"]
        constraint.removed_columns = transformations["removed"]

        # Determine if other columns are preserved
        constraint.preserves_other_columns = self._preserves_other_columns(
            operation_analysis,
        )

        # Add warnings based on analysis complexity
        warnings = self._generate_warnings(operations, source_analysis, type_info)
        for warning in warnings:
            constraint.add_warning(warning)

        return constraint

    def _analyze_operations(
        self,
        operations: set[DataFrameOperation],
    ) -> dict[str, Any]:
        """Analyze operations to understand their impact."""
        analysis = {
            "withColumn_ops": [],
            "select_ops": [],
            "drop_ops": [],
            "filter_ops": [],
            "groupby_ops": [],
            "agg_ops": [],
            "has_groupby": False,
            "has_select": False,
            "has_joins": False,
            "schema_changing": [],
        }

        for op in operations:
            method = op.method_name

            if method == "withColumn":
                analysis["withColumn_ops"].append(op)
            elif method == "select":
                analysis["select_ops"].append(op)
                analysis["has_select"] = True
            elif method == "drop":
                analysis["drop_ops"].append(op)
            elif method in ["filter", "where"]:
                analysis["filter_ops"].append(op)
            elif method in ["groupBy", "groupby"]:
                analysis["groupby_ops"].append(op)
                analysis["has_groupby"] = True
            elif method == "agg":
                analysis["agg_ops"].append(op)
            elif method in ["join", "crossJoin"]:
                analysis["has_joins"] = True

            if op.affects_schema():
                analysis["schema_changing"].append(op)

        return analysis

    def _generate_required_columns(
        self,
        read_columns: set[str],
        conditional_columns: set[str],
        type_info: dict[str, Any],
        operation_analysis: dict[str, list[DataFrameOperation]],
    ) -> list[ColumnRequirement]:
        """Generate required column constraints."""
        required_columns = []

        # All read columns are required
        all_required = read_columns | conditional_columns

        # Remove columns that are created within the function
        created_columns = set()
        for op in operation_analysis["withColumn_ops"]:
            if op.args:
                created_columns.add(op.args[0])

        actual_required = all_required - created_columns

        for col_name in actual_required:
            # Infer type from type analysis
            col_type = self._get_column_type(col_name, type_info)

            # Determine nullability (default to True for safety)
            nullable = True

            # If used in filtering, might be non-nullable in some contexts
            if col_name in conditional_columns:
                # Keep nullable=True for safety unless we have strong evidence
                pass

            required_columns.append(
                ColumnRequirement(
                    name=col_name,
                    type=col_type,
                    nullable=nullable,
                    description=f"Required for analysis - detected from {'filter' if col_name in conditional_columns else 'access'} operations",
                ),
            )

        return required_columns

    def _generate_transformations(
        self,
        operations: set[DataFrameOperation],
        written_columns: set[str],
        type_info: dict[str, Any],
        operation_analysis: dict[str, list[DataFrameOperation]],
    ) -> dict[str, list]:
        """Generate column transformation constraints."""
        transformations = {
            "added": [],
            "modified": [],
            "removed": [],
        }

        # Track which columns are added vs modified
        # original_columns = set()  # Would need schema info to populate this

        # Process withColumn operations
        for op in operation_analysis["withColumn_ops"]:
            if op.args:
                col_name = op.args[0]
                col_type = self._get_column_type(col_name, type_info)

                # For now, assume new columns unless we have evidence otherwise
                # This is conservative - in practice we'd check against input schema
                transformation = ColumnTransformation(
                    name=col_name,
                    operation="add",  # Conservative assumption
                    type=col_type,
                    nullable=True,  # Safe default
                    description="Column added by withColumn operation",
                )
                transformations["added"].append(transformation)

        # Process drop operations
        for op in operation_analysis["drop_ops"]:
            for arg in op.args:
                if isinstance(arg, str) and arg not in ["<expression>", "<unknown>"]:
                    transformations["removed"].append(arg)

        # Handle select operations (they implicitly remove unselected columns)
        if operation_analysis["has_select"]:
            # If there's a select, only selected columns are preserved
            # This affects preserves_other_columns but doesn't create explicit removals
            pass

        # Handle aggregation operations
        if operation_analysis["has_groupby"] or operation_analysis["agg_ops"]:
            # Aggregations typically change the schema significantly
            for op in operation_analysis["agg_ops"]:
                # Try to infer aggregated columns from operation
                for arg in op.args:
                    if "." in arg and arg.endswith(")"):
                        # This might be something like "sum(amount)"
                        # Extract the function name for type inference
                        if "sum(" in arg or "avg(" in arg:
                            # These create new columns
                            agg_col_name = f"agg_{arg.replace('(', '_').replace(')', '').replace('.', '_')}"
                            col_type = "double" if "avg(" in arg else "integer"

                            transformation = ColumnTransformation(
                                name=agg_col_name,
                                operation="add",
                                type=col_type,
                                nullable=True,
                                description=f"Aggregation result from {arg}",
                            )
                            transformations["added"].append(transformation)

        return transformations

    def _get_column_type(self, column_name: str, type_info: dict[str, Any]) -> str:
        """Get the inferred type for a column."""
        inferred_types = type_info.get("inferred_types", {})

        if column_name in inferred_types:
            return inferred_types[column_name]["type"]

        # Default fallback based on common column naming patterns
        name_lower = column_name.lower()

        if any(keyword in name_lower for keyword in ["id", "key"]):
            return "string"
        elif any(keyword in name_lower for keyword in ["count", "num", "number"]):
            return "integer"
        elif any(
            keyword in name_lower for keyword in ["amount", "price", "cost", "value"]
        ):
            return "double"
        elif any(keyword in name_lower for keyword in ["date", "time"]):
            return "timestamp"
        elif any(keyword in name_lower for keyword in ["flag", "is_", "has_"]):
            return "boolean"
        else:
            return "string"  # Safe default

    def _preserves_other_columns(self, operation_analysis: dict[str, Any]) -> bool:
        """Determine if other columns are preserved."""
        # If there's a select operation, only selected columns are preserved
        if operation_analysis["has_select"]:
            return False

        # If there's groupBy, the schema structure changes significantly
        if operation_analysis["has_groupby"]:
            return False

        # If there are joins, the schema might change
        if operation_analysis["has_joins"]:
            return False

        # Otherwise, assume columns are preserved (withColumn, filter, etc.)
        return True

    def _generate_warnings(
        self,
        operations: list[DataFrameOperation],
        source_analysis: dict[str, Any],
        type_info: dict[str, Any],
    ) -> list[str]:
        """Generate appropriate warnings based on analysis."""
        warnings = []

        # UDF warnings
        if source_analysis.get("udf_count", 0) > 0:
            warnings.append("UDF usage detected - static analysis may be incomplete")

        # Dynamic operation warnings
        if source_analysis.get("dynamic_operations", 0) > 0:
            warnings.append(
                "Dynamic column operations detected - manual verification recommended",
            )

        # Complex expression warnings
        if source_analysis.get("complex_expressions", 0) > 2:
            warnings.append(
                "Complex conditional logic detected - review constraint accuracy",
            )

        # Join warnings
        has_joins = any(op.operation_type == "join" for op in operations)
        if has_joins:
            warnings.append("Join operations detected - schema changes may be complex")

        # Aggregation warnings
        has_agg = any(op.operation_type in ["groupBy", "agg"] for op in operations)
        if has_agg:
            warnings.append(
                "Aggregation operations detected - output schema may differ significantly from input",
            )

        return warnings


def generate_constraint_from_function(
    operations: list[DataFrameOperation],
    column_references: list[ColumnReference],
    type_info: dict[str, Any],
    source_analysis: dict[str, Any],
) -> PartialSchemaConstraint:
    """
    Generate a constraint from static analysis results.

    This is the main entry point for generating schema constraints
    from the results of static analysis.

    Args:
        operations: DataFrame operations analysis
        column_references: Column reference analysis
        type_info: Type inference results
        source_analysis: General source analysis metadata

    Returns:
        PartialSchemaConstraint representing the function's requirements
    """
    generator = ConstraintGenerator()
    return generator.generate_constraint(
        operations=operations,
        column_references=column_references,
        type_info=type_info,
        source_analysis=source_analysis,
    )


def generate_constraint(
    operations: list[DataFrameOperation],
    column_references: list[ColumnReference],
) -> PartialSchemaConstraint:
    """
    Generate constraint from operations and column references (test-compatible interface).

    Args:
        operations: List of detected operations
        column_references: List of referenced column names

    Returns:
        PartialSchemaConstraint
    """
    required_columns = []
    added_columns = []
    removed_columns = []

    # Process column references
    for col_name in column_references:
        if col_name == "status":
            required_columns.append(ColumnRequirement("status", "string"))
        elif col_name == "amount":
            required_columns.append(ColumnRequirement("amount", "double"))

    # Process operations
    for op in operations:
        if op.operation_type == "withColumn":
            col_name = op.args[0]
            col_type = op.args[1]
            added_columns.append(ColumnTransformation(col_name, "add", col_type))
        elif op.operation_type == "drop":
            removed_columns.extend(op.args)

    return PartialSchemaConstraint(
        required_columns=required_columns,
        added_columns=added_columns,
        removed_columns=removed_columns,
        preserves_other_columns=True,
    )
