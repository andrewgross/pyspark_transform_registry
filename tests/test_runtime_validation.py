"""
Tests for the runtime validation engine.

This module tests the runtime validation of PySpark DataFrames against
schema constraints, ensuring that validation works correctly for various
scenarios and edge cases.
"""

import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    IntegerType,
    DoubleType,
    TimestampType,
)

from pyspark_transform_registry.runtime_validation import (
    RuntimeValidator,
    validate_dataframe_against_constraint,
    check_dataframe_compatibility,
)
from pyspark_transform_registry.schema_constraints import (
    PartialSchemaConstraint,
    ColumnRequirement,
    ColumnTransformation,
    ValidationResult,
)


@pytest.fixture(scope="session")
def spark():
    """Create Spark session for testing."""
    return SparkSession.builder.appName("RuntimeValidationTests").getOrCreate()


class TestRuntimeValidator:
    """Test the RuntimeValidator class."""

    def test_validator_initialization(self):
        """Test RuntimeValidator initialization."""
        # Default initialization
        validator = RuntimeValidator()
        assert validator.strict_mode is False

        # Strict mode initialization
        strict_validator = RuntimeValidator(strict_mode=True)
        assert strict_validator.strict_mode is True

    def test_validate_simple_required_columns(self, spark):
        """Test validation of simple required columns."""
        # Create test DataFrame
        schema = StructType(
            [
                StructField("customer_id", StringType(), True),
                StructField("amount", DoubleType(), True),
            ],
        )
        df = spark.createDataFrame([("cust1", 100.0)], schema)

        # Create constraint
        constraint = PartialSchemaConstraint(
            required_columns=[
                ColumnRequirement("customer_id", "string"),
                ColumnRequirement("amount", "double"),
            ],
        )

        # Validate
        validator = RuntimeValidator()
        result = validator.validate_dataframe(df, constraint)

        assert result.is_valid is True
        assert len(result.get_error_messages()) == 0

    def test_validate_missing_required_column(self, spark):
        """Test validation when required column is missing."""
        # Create test DataFrame missing 'amount' column
        schema = StructType(
            [
                StructField("customer_id", StringType(), True),
            ],
        )
        df = spark.createDataFrame([("cust1",)], schema)

        # Create constraint requiring 'amount'
        constraint = PartialSchemaConstraint(
            required_columns=[
                ColumnRequirement("customer_id", "string"),
                ColumnRequirement("amount", "double"),
            ],
        )

        # Validate
        validator = RuntimeValidator()
        result = validator.validate_dataframe(df, constraint)

        assert result.is_valid is False
        errors = result.get_error_messages()
        assert len(errors) == 1
        assert "Required column 'amount' is missing" in errors[0]

    def test_validate_incorrect_column_type(self, spark):
        """Test validation when column has incorrect type."""
        # Create test DataFrame with string where double expected
        schema = StructType(
            [
                StructField("customer_id", StringType(), True),
                StructField("amount", StringType(), True),  # Should be double
            ],
        )
        df = spark.createDataFrame([("cust1", "100.0")], schema)

        # Create constraint
        constraint = PartialSchemaConstraint(
            required_columns=[
                ColumnRequirement("customer_id", "string"),
                ColumnRequirement("amount", "double"),
            ],
        )

        # Validate
        validator = RuntimeValidator()
        result = validator.validate_dataframe(df, constraint)

        assert result.is_valid is False
        errors = result.get_error_messages()
        assert len(errors) == 1
        assert "Column 'amount' has incorrect type" in errors[0]

    def test_validate_optional_columns_present(self, spark):
        """Test validation of optional columns when present."""
        # Create test DataFrame with optional column
        schema = StructType(
            [
                StructField("customer_id", StringType(), True),
                StructField("phone", StringType(), True),
            ],
        )
        df = spark.createDataFrame([("cust1", "555-1234")], schema)

        # Create constraint with optional column
        constraint = PartialSchemaConstraint(
            required_columns=[
                ColumnRequirement("customer_id", "string"),
            ],
            optional_columns=[
                ColumnRequirement("phone", "string"),
            ],
        )

        # Validate
        validator = RuntimeValidator()
        result = validator.validate_dataframe(df, constraint)

        assert result.is_valid is True
        assert len(result.get_error_messages()) == 0

    def test_validate_optional_columns_missing(self, spark):
        """Test validation when optional columns are missing."""
        # Create test DataFrame without optional column
        schema = StructType(
            [
                StructField("customer_id", StringType(), True),
            ],
        )
        df = spark.createDataFrame([("cust1",)], schema)

        # Create constraint with optional column
        constraint = PartialSchemaConstraint(
            required_columns=[
                ColumnRequirement("customer_id", "string"),
            ],
            optional_columns=[
                ColumnRequirement("phone", "string"),
            ],
        )

        # Validate
        validator = RuntimeValidator()
        result = validator.validate_dataframe(df, constraint)

        assert result.is_valid is True  # Optional columns can be missing
        assert len(result.get_error_messages()) == 0

    def test_validate_optional_column_wrong_type(self, spark):
        """Test validation when optional column has wrong type."""
        # Create test DataFrame with wrong type for optional column
        schema = StructType(
            [
                StructField("customer_id", StringType(), True),
                StructField("phone", IntegerType(), True),  # Should be string
            ],
        )
        df = spark.createDataFrame([("cust1", 5551234)], schema)

        # Create constraint with optional column
        constraint = PartialSchemaConstraint(
            required_columns=[
                ColumnRequirement("customer_id", "string"),
            ],
            optional_columns=[
                ColumnRequirement("phone", "string"),
            ],
        )

        # Validate
        validator = RuntimeValidator()
        result = validator.validate_dataframe(df, constraint)

        assert result.is_valid is False
        errors = result.get_error_messages()
        assert len(errors) == 1
        assert "Optional column 'phone' has incorrect type" in errors[0]

    def test_validate_unexpected_columns_strict(self, spark):
        """Test validation of unexpected columns when not preserving others."""
        # Create test DataFrame with extra column
        schema = StructType(
            [
                StructField("customer_id", StringType(), True),
                StructField("extra_column", StringType(), True),
            ],
        )
        df = spark.createDataFrame([("cust1", "extra")], schema)

        # Create constraint that doesn't preserve other columns
        constraint = PartialSchemaConstraint(
            required_columns=[
                ColumnRequirement("customer_id", "string"),
            ],
            preserves_other_columns=False,
        )

        # Validate
        validator = RuntimeValidator()
        result = validator.validate_dataframe(df, constraint)

        assert result.is_valid is True  # Warnings don't make invalid
        warnings = result.get_warning_messages()
        assert len(warnings) >= 1
        assert any("Unexpected column 'extra_column'" in w for w in warnings)

    def test_validate_preserves_other_columns(self, spark):
        """Test validation when preserving other columns."""
        # Create test DataFrame with extra column
        schema = StructType(
            [
                StructField("customer_id", StringType(), True),
                StructField("extra_column", StringType(), True),
            ],
        )
        df = spark.createDataFrame([("cust1", "extra")], schema)

        # Create constraint that preserves other columns
        constraint = PartialSchemaConstraint(
            required_columns=[
                ColumnRequirement("customer_id", "string"),
            ],
            preserves_other_columns=True,
        )

        # Validate
        validator = RuntimeValidator()
        result = validator.validate_dataframe(df, constraint)

        assert result.is_valid is True
        # Should not warn about extra columns when preserving others

    def test_validate_numeric_type_compatibility(self, spark):
        """Test numeric type compatibility (integer vs double)."""
        # Create test DataFrame with integer where double expected
        schema = StructType(
            [
                StructField("amount", IntegerType(), True),
            ],
        )
        df = spark.createDataFrame([(100,)], schema)

        # Create constraint expecting double
        constraint = PartialSchemaConstraint(
            required_columns=[
                ColumnRequirement("amount", "double"),
            ],
        )

        # Validate - should be compatible
        validator = RuntimeValidator()
        result = validator.validate_dataframe(df, constraint)

        assert result.is_valid is True
        assert len(result.get_error_messages()) == 0

    def test_validate_analysis_warnings(self, spark):
        """Test that analysis warnings are included in validation."""
        # Create simple DataFrame
        schema = StructType(
            [
                StructField("customer_id", StringType(), True),
            ],
        )
        df = spark.createDataFrame([("cust1",)], schema)

        # Create constraint with analysis warnings
        constraint = PartialSchemaConstraint(
            required_columns=[
                ColumnRequirement("customer_id", "string"),
            ],
            warnings=["UDF detected", "Complex logic found"],
        )

        # Validate
        validator = RuntimeValidator()
        result = validator.validate_dataframe(df, constraint)

        assert result.is_valid is True
        warnings = result.get_warning_messages()
        assert any("UDF detected" in w for w in warnings)
        assert any("Complex logic found" in w for w in warnings)


class TestUtilityFunctions:
    """Test utility functions for runtime validation."""

    def test_validate_dataframe_against_constraint(self, spark):
        """Test the convenience function."""
        # Create test DataFrame
        schema = StructType(
            [
                StructField("customer_id", StringType(), True),
            ],
        )
        df = spark.createDataFrame([("cust1",)], schema)

        # Create constraint
        constraint = PartialSchemaConstraint(
            required_columns=[
                ColumnRequirement("customer_id", "string"),
            ],
        )

        # Validate using convenience function
        result = validate_dataframe_against_constraint(df, constraint)

        assert isinstance(result, ValidationResult)
        assert result.is_valid is True

    def test_check_dataframe_compatibility(self, spark):
        """Test the quick compatibility check function."""
        # Create test DataFrame
        schema = StructType(
            [
                StructField("customer_id", StringType(), True),
            ],
        )
        df = spark.createDataFrame([("cust1",)], schema)

        # Create valid constraint
        valid_constraint = PartialSchemaConstraint(
            required_columns=[
                ColumnRequirement("customer_id", "string"),
            ],
        )

        # Create invalid constraint
        invalid_constraint = PartialSchemaConstraint(
            required_columns=[
                ColumnRequirement("missing_column", "string"),
            ],
        )

        # Test compatibility
        assert check_dataframe_compatibility(df, valid_constraint) is True
        assert check_dataframe_compatibility(df, invalid_constraint) is False


class TestComplexValidationScenarios:
    """Test complex validation scenarios."""

    def test_validate_complex_constraint(self, spark):
        """Test validation with a complex constraint."""
        from datetime import datetime

        # Create test DataFrame
        schema = StructType(
            [
                StructField("customer_id", StringType(), False),
                StructField("amount", DoubleType(), True),
                StructField("category", StringType(), True),
                StructField("created_at", TimestampType(), False),
            ],
        )
        df = spark.createDataFrame(
            [("cust1", 100.0, "premium", datetime(2023, 1, 1))],
            schema,
        )

        # Create complex constraint
        constraint = PartialSchemaConstraint(
            required_columns=[
                ColumnRequirement("customer_id", "string", nullable=False),
                ColumnRequirement("amount", "double"),
            ],
            optional_columns=[
                ColumnRequirement("category", "string"),
            ],
            added_columns=[
                ColumnTransformation("processed_at", "add", "timestamp"),
            ],
            preserves_other_columns=True,
        )

        # Validate
        validator = RuntimeValidator()
        result = validator.validate_dataframe(df, constraint)

        assert result.is_valid is True

    def test_validate_nullability_warnings(self, spark):
        """Test nullability validation warnings."""
        # Create DataFrame with nullable column
        schema = StructType(
            [
                StructField("customer_id", StringType(), True),  # Nullable
            ],
        )
        df = spark.createDataFrame([("cust1",)], schema)

        # Create constraint expecting non-nullable
        constraint = PartialSchemaConstraint(
            required_columns=[
                ColumnRequirement("customer_id", "string", nullable=False),
            ],
        )

        # Validate
        validator = RuntimeValidator()
        result = validator.validate_dataframe(df, constraint)

        assert result.is_valid is True  # Still valid, just warning
        warnings = result.get_warning_messages()
        assert any(
            "allows nulls but constraint expects non-nullable" in w for w in warnings
        )

    def test_validate_empty_dataframe(self, spark):
        """Test validation of empty DataFrame."""
        # Create empty DataFrame
        schema = StructType(
            [
                StructField("customer_id", StringType(), True),
            ],
        )
        df = spark.createDataFrame([], schema)

        # Create constraint
        constraint = PartialSchemaConstraint(
            required_columns=[
                ColumnRequirement("customer_id", "string"),
            ],
        )

        # Validate - should work for schema validation
        validator = RuntimeValidator()
        result = validator.validate_dataframe(df, constraint)

        assert result.is_valid is True

    def test_validate_empty_constraint(self, spark):
        """Test validation against empty constraint."""
        # Create test DataFrame
        schema = StructType(
            [
                StructField("any_column", StringType(), True),
            ],
        )
        df = spark.createDataFrame([("value",)], schema)

        # Create empty constraint (no requirements)
        constraint = PartialSchemaConstraint()

        # Validate - should be valid
        validator = RuntimeValidator()
        result = validator.validate_dataframe(df, constraint)

        assert result.is_valid is True
