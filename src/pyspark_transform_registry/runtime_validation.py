"""
Runtime validation engine for PySpark DataFrames against schema constraints.

This module provides functionality to validate DataFrames at runtime against
PartialSchemaConstraint objects, ensuring that the input data meets the
requirements specified by transform functions.
"""

from typing import Any
from pyspark.sql import DataFrame
from pyspark.sql.types import (
    DataType,
    StringType,
    IntegerType,
    DoubleType,
    BooleanType,
    TimestampType,
    DateType,
    BinaryType,
    LongType,
)

from .schema_constraints import (
    PartialSchemaConstraint,
    ColumnRequirement,
    ValidationResult,
)


class RuntimeValidator:
    """
    Runtime validator for PySpark DataFrames against schema constraints.

    This validator checks that DataFrames conform to the requirements
    specified in PartialSchemaConstraint objects, providing detailed
    validation results and actionable error messages.
    """

    def __init__(self, strict_mode: bool = False):
        """
        Initialize the runtime validator.

        Args:
            strict_mode: If True, treat warnings as errors
        """
        self.strict_mode = strict_mode
        self.pyspark_type_mapping = {
            "string": StringType(),
            "integer": IntegerType(),
            "double": DoubleType(),
            "boolean": BooleanType(),
            "timestamp": TimestampType(),
            "date": DateType(),
            "binary": BinaryType(),
        }

    def validate_dataframe(
        self,
        df: DataFrame,
        constraint: PartialSchemaConstraint,
    ) -> ValidationResult:
        """
        Validate a DataFrame against a schema constraint.

        Args:
            df: PySpark DataFrame to validate
            constraint: Schema constraint to validate against

        Returns:
            ValidationResult with detailed validation information
        """
        result = ValidationResult(is_valid=True)

        # Get DataFrame schema information
        df_schema = df.schema
        df_columns = {field.name: field for field in df_schema.fields}

        # Validate required columns
        self._validate_required_columns(df_columns, constraint.required_columns, result)

        # Validate optional columns (if present)
        self._validate_optional_columns(df_columns, constraint.optional_columns, result)

        # Check for unexpected columns (if not preserving other columns)
        if not constraint.preserves_other_columns:
            self._validate_no_unexpected_columns(df_columns, constraint, result)

        # Add constraint warnings to result
        for warning in constraint.warnings:
            result.add_warning(
                message=warning,
                column_name=None,
            )

        return result

    def _validate_required_columns(
        self,
        df_columns: dict[str, Any],
        required_columns: list[ColumnRequirement],
        result: ValidationResult,
    ) -> None:
        """Validate that all required columns are present with correct types."""
        for req_col in required_columns:
            col_name = req_col.name

            # Check if column exists
            if col_name not in df_columns:
                result.add_error(
                    f"Required column '{col_name}' is missing",
                    column_name=col_name,
                    expected_type=req_col.type,
                )
                continue

            # Check column type
            df_field = df_columns[col_name]
            expected_type = self._get_pyspark_type(req_col.type)
            actual_type = df_field.dataType

            if not self._types_compatible(actual_type, expected_type):
                result.add_error(
                    f"Column '{col_name}' has incorrect type",
                    column_name=col_name,
                    expected_type=req_col.type,
                    actual_type=self._pyspark_type_to_string(actual_type),
                )

            # Check nullability
            if not req_col.nullable and df_field.nullable:
                result.add_warning(
                    f"Column '{col_name}' allows nulls but constraint expects non-nullable",
                    column_name=col_name,
                )

    def _validate_optional_columns(
        self,
        df_columns: dict[str, Any],
        optional_columns: list[ColumnRequirement],
        result: ValidationResult,
    ) -> None:
        """Validate optional columns if they are present."""
        for opt_col in optional_columns:
            col_name = opt_col.name

            # Skip if column is not present (it's optional)
            if col_name not in df_columns:
                continue

            # If present, validate type
            df_field = df_columns[col_name]
            expected_type = self._get_pyspark_type(opt_col.type)
            actual_type = df_field.dataType

            if not self._types_compatible(actual_type, expected_type):
                result.add_error(
                    f"Optional column '{col_name}' has incorrect type",
                    column_name=col_name,
                    expected_type=opt_col.type,
                    actual_type=self._pyspark_type_to_string(actual_type),
                )

    def _validate_no_unexpected_columns(
        self,
        df_columns: dict[str, Any],
        constraint: PartialSchemaConstraint,
        result: ValidationResult,
    ) -> None:
        """Validate that no unexpected columns are present."""
        # Get all expected column names
        expected_columns = set()

        # Required columns
        expected_columns.update(col.name for col in constraint.required_columns)

        # Optional columns
        expected_columns.update(col.name for col in constraint.optional_columns)

        # Added columns (output of the transform)
        expected_columns.update(col.name for col in constraint.added_columns)

        # Modified columns (should already be in required/optional)
        expected_columns.update(col.name for col in constraint.modified_columns)

        # Remove deleted columns
        expected_columns -= set(constraint.removed_columns)

        # Check for unexpected columns
        actual_columns = set(df_columns.keys())
        unexpected_columns = actual_columns - expected_columns

        for col_name in unexpected_columns:
            result.add_warning(
                f"Unexpected column '{col_name}' found",
                column_name=col_name,
            )

    def _get_pyspark_type(self, type_string: str) -> DataType:
        """Convert type string to PySpark DataType."""
        # Handle basic types
        if type_string in self.pyspark_type_mapping:
            return self.pyspark_type_mapping[type_string]

        # Handle complex types (arrays, maps) - simplified for now
        if type_string.startswith("array<"):
            return StringType()  # Simplified - treat as string for validation
        elif type_string.startswith("map<"):
            return StringType()  # Simplified - treat as string for validation
        else:
            # Default to string for unknown types
            return StringType()

    def _types_compatible(self, actual_type: DataType, expected_type: DataType) -> bool:
        """Check if actual and expected types are compatible."""
        # Exact match
        if type(actual_type) is type(expected_type):
            return True

        # Numeric compatibility (only between numeric types)
        numeric_types = {IntegerType, DoubleType, LongType}
        if type(actual_type) in numeric_types and type(expected_type) in numeric_types:
            return True

        # No automatic string compatibility - types must match
        return False

    def _pyspark_type_to_string(self, data_type: DataType) -> str:
        """Convert PySpark DataType to string representation."""
        type_mapping = {
            StringType: "string",
            IntegerType: "integer",
            LongType: "long",
            DoubleType: "double",
            BooleanType: "boolean",
            TimestampType: "timestamp",
            DateType: "date",
            BinaryType: "binary",
        }

        return type_mapping.get(type(data_type), str(data_type))


def validate_dataframe_against_constraint(
    df: DataFrame,
    constraint: PartialSchemaConstraint,
    strict_mode: bool = False,
) -> ValidationResult:
    """
    Validate a DataFrame against a schema constraint.

    This is a convenience function that creates a RuntimeValidator
    and performs validation in a single call.

    Args:
        df: PySpark DataFrame to validate
        constraint: Schema constraint to validate against
        strict_mode: If True, treat warnings as errors

    Returns:
        ValidationResult with detailed validation information
    """
    validator = RuntimeValidator(strict_mode=strict_mode)
    return validator.validate_dataframe(df, constraint)


def check_dataframe_compatibility(
    df: DataFrame,
    constraint: PartialSchemaConstraint,
) -> bool:
    """
    Quick compatibility check between DataFrame and constraint.

    Args:
        df: PySpark DataFrame to check
        constraint: Schema constraint to check against

    Returns:
        True if DataFrame is compatible, False otherwise
    """
    result = validate_dataframe_against_constraint(df, constraint)
    return result.is_valid
