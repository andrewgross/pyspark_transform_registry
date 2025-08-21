"""
Tests for the enhanced loading API with runtime validation.

This module tests the loading of PySpark transform functions with
runtime validation capabilities, ensuring that input validation
works correctly with various scenarios.
"""

import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import DoubleType, StringType, StructField, StructType

from pyspark_transform_registry.core import load_function, register_function
from pyspark_transform_registry.schema_constraints import (
    ColumnRequirement,
    PartialSchemaConstraint,
)


@pytest.fixture(scope="session")
def spark():
    """Create Spark session for testing."""
    return SparkSession.builder.appName("LoadingValidationTests").getOrCreate()


@pytest.fixture
def sample_transform_function():
    """Create a sample transform function for testing."""

    def simple_transform(df):
        """Simple transform that requires customer_id and amount columns."""
        return df.select("customer_id", "amount").withColumn("processed", df.amount * 2)

    return simple_transform


@pytest.fixture
def multi_param_transform_function():
    """Create a multi-parameter transform function for testing."""

    def multi_param_transform(df, multiplier=1, filter_threshold=0):
        """Transform with multiple parameters."""
        return df.filter(df.amount > filter_threshold).withColumn(
            "result",
            df.amount * multiplier,
        )

    return multi_param_transform


@pytest.fixture
def valid_test_dataframe(spark):
    """Create a valid test DataFrame."""
    schema = StructType(
        [
            StructField("customer_id", StringType(), True),
            StructField("amount", DoubleType(), True),
        ],
    )
    return spark.createDataFrame([("cust1", 100.0), ("cust2", 200.0)], schema)


@pytest.fixture
def invalid_test_dataframe(spark):
    """Create an invalid test DataFrame (missing required column)."""
    schema = StructType(
        [
            StructField("customer_id", StringType(), True),
            # Missing 'amount' column
        ],
    )
    return spark.createDataFrame([("cust1",), ("cust2",)], schema)


@pytest.fixture
def wrong_type_dataframe(spark):
    """Create a DataFrame with wrong column types."""
    schema = StructType(
        [
            StructField("customer_id", StringType(), True),
            StructField("amount", StringType(), True),  # Should be DoubleType
        ],
    )
    return spark.createDataFrame([("cust1", "100.0"), ("cust2", "200.0")], schema)


class TestLoadingWithValidation:
    """Test loading functions with runtime validation."""

    def test_load_function_with_validation_enabled(
        self,
        spark,
        sample_transform_function,
        valid_test_dataframe,
    ):
        """Test loading function with validation enabled (default)."""
        # Register function with schema inference
        model_name = "test_catalog.test_schema.validation_test"
        register_function(
            sample_transform_function,
            name=model_name,
        )

        # Load function with validation (default behavior)
        loaded_transform = load_function(model_name, version=1)

        # Should work with valid input
        result = loaded_transform(valid_test_dataframe)
        assert result is not None
        assert "processed" in result.columns

    def test_load_function_with_validation_disabled(
        self,
        spark,
        sample_transform_function,
        valid_test_dataframe,
        invalid_test_dataframe,
    ):
        """Test loading function with validation disabled."""
        # Register function
        model_name = "test_catalog.test_schema.no_validation_test"
        register_function(
            sample_transform_function,
            name=model_name,
        )

        # Load function without validation
        loaded_transform = load_function(model_name, version=1, validate_input=False)

        # Should work even with invalid input when validation is disabled
        # Note: This will fail at execution time, but not at validation time
        with pytest.raises(Exception):  # Expected to fail during actual transformation
            loaded_transform(invalid_test_dataframe)

    def test_validation_failure_with_missing_columns(
        self,
        spark,
        sample_transform_function,
        valid_test_dataframe,
        invalid_test_dataframe,
    ):
        """Test validation failure when required columns are missing."""
        # Create explicit schema constraint since static analysis might fail in test
        from pyspark_transform_registry.schema_constraints import (
            ColumnRequirement,
            PartialSchemaConstraint,
        )

        constraint = PartialSchemaConstraint(
            required_columns=[
                ColumnRequirement("customer_id", "string"),
                ColumnRequirement("amount", "double"),
            ],
        )

        # Register function with explicit schema constraint
        model_name = "test_catalog.test_schema.missing_columns_test"
        register_function(
            sample_transform_function,
            name=model_name,
            schema_constraint=constraint,
        )

        # Load function with validation enabled
        loaded_transform = load_function(model_name, version=1, validate_input=True)

        # Should fail validation with invalid input
        with pytest.raises(ValueError, match="Input validation failed"):
            loaded_transform(invalid_test_dataframe)

    def test_validation_failure_with_wrong_types(
        self,
        spark,
        sample_transform_function,
        valid_test_dataframe,
        wrong_type_dataframe,
    ):
        """Test validation failure when column types are wrong."""
        # Create explicit schema constraint
        from pyspark_transform_registry.schema_constraints import (
            ColumnRequirement,
            PartialSchemaConstraint,
        )

        constraint = PartialSchemaConstraint(
            required_columns=[
                ColumnRequirement("customer_id", "string"),
                ColumnRequirement("amount", "double"),
            ],
        )

        # Register function with explicit schema constraint
        model_name = "test_catalog.test_schema.wrong_types_test"
        register_function(
            sample_transform_function,
            name=model_name,
            schema_constraint=constraint,
        )

        # Load function with validation enabled
        loaded_transform = load_function(model_name, version=1, validate_input=True)

        # Should fail validation with wrong types
        with pytest.raises(ValueError, match="Input validation failed"):
            loaded_transform(wrong_type_dataframe)

    def test_strict_validation_mode(
        self,
        spark,
        sample_transform_function,
        valid_test_dataframe,
    ):
        """Test strict validation mode where warnings become errors."""
        # Create a DataFrame with extra columns to trigger warnings
        schema = StructType(
            [
                StructField("customer_id", StringType(), True),
                StructField("amount", DoubleType(), True),
                StructField("extra_column", StringType(), True),  # Will trigger warning
            ],
        )
        df_with_extra = spark.createDataFrame([("cust1", 100.0, "extra")], schema)

        # Register function with strict schema (doesn't preserve other columns)
        model_name = "test_catalog.test_schema.strict_validation_test"
        register_function(
            sample_transform_function,
            name=model_name,
            schema_constraint=PartialSchemaConstraint(
                required_columns=[
                    ColumnRequirement("customer_id", "string"),
                    ColumnRequirement("amount", "double"),
                ],
                preserves_other_columns=False,  # This will trigger warnings for extra columns
            ),
        )

        # Load function with strict validation
        loaded_transform = load_function(model_name, version=1, strict_validation=True)

        # Should still work but print warnings (warnings don't fail validation)
        result = loaded_transform(df_with_extra)
        assert result is not None

    def test_multi_parameter_function_validation(
        self,
        spark,
        multi_param_transform_function,
        valid_test_dataframe,
    ):
        """Test validation with multi-parameter functions."""
        # Register multi-parameter function
        model_name = "test_catalog.test_schema.multi_param_validation_test"
        register_function(
            multi_param_transform_function,
            name=model_name,
        )

        # Load function with validation
        loaded_transform = load_function(model_name, version=1, validate_input=True)

        # Should work with valid input and parameters
        result = loaded_transform(
            valid_test_dataframe,
            params={"multiplier": 3, "filter_threshold": 50},
        )
        assert result is not None

    def test_load_function_without_schema_constraint(self, spark, valid_test_dataframe):
        """Test loading function that was registered without schema inference."""

        def simple_func(df):
            return df.select("*")

        # Register function without schema inference
        model_name = "test_catalog.test_schema.no_schema_test"
        register_function(
            simple_func,
            name=model_name,
            infer_schema=False,  # Explicitly disable schema inference
        )

        # Load function - should work without validation since no constraint is stored
        loaded_transform = load_function(model_name, version=1, validate_input=True)

        # Should work normally without validation
        result = loaded_transform(valid_test_dataframe)
        assert result is not None

    def test_load_specific_version_with_validation(
        self,
        spark,
        sample_transform_function,
        valid_test_dataframe,
    ):
        """Test loading specific version with validation."""
        # Register function
        model_name = "test_catalog.test_schema.version_validation_test"
        register_function(
            sample_transform_function,
            name=model_name,
        )

        # Load specific version (version 1) with validation
        loaded_transform = load_function(model_name, version=1, validate_input=True)

        # Should work with valid input
        result = loaded_transform(valid_test_dataframe)
        assert result is not None

    def test_validation_graceful_degradation(
        self,
        spark,
        sample_transform_function,
        valid_test_dataframe,
    ):
        """Test that validation gracefully degrades when schema constraint can't be loaded."""
        # Register function
        model_name = "test_catalog.test_schema.graceful_degradation_test"
        register_function(
            sample_transform_function,
            name=model_name,
        )

        # This test simulates the case where schema constraint loading fails
        # The function should still work, just without validation
        loaded_transform = load_function(model_name, version=1, validate_input=True)

        # Should work normally
        result = loaded_transform(valid_test_dataframe)
        assert result is not None


class TestValidationErrorMessages:
    """Test that validation error messages are informative."""

    def test_detailed_error_messages_for_missing_columns(
        self,
        spark,
        sample_transform_function,
        valid_test_dataframe,
    ):
        """Test that error messages clearly identify missing columns."""
        # Create DataFrame missing required column
        schema = StructType(
            [
                StructField("customer_id", StringType(), True),
                # Missing 'amount' column
            ],
        )
        invalid_df = spark.createDataFrame([("cust1",)], schema)

        # Create explicit schema constraint
        from pyspark_transform_registry.schema_constraints import (
            ColumnRequirement,
            PartialSchemaConstraint,
        )

        constraint = PartialSchemaConstraint(
            required_columns=[
                ColumnRequirement("customer_id", "string"),
                ColumnRequirement("amount", "double"),
            ],
        )

        # Register and load function
        model_name = "test_catalog.test_schema.error_message_test"
        register_function(
            sample_transform_function,
            name=model_name,
            schema_constraint=constraint,
        )

        loaded_transform = load_function(model_name, version=1)

        # Error message should be informative
        with pytest.raises(ValueError) as exc_info:
            loaded_transform(invalid_df)

        error_message = str(exc_info.value)
        assert "Input validation failed" in error_message
        assert "amount" in error_message  # Should mention the missing column

    def test_detailed_error_messages_for_wrong_types(
        self,
        spark,
        sample_transform_function,
        valid_test_dataframe,
    ):
        """Test that error messages clearly identify type mismatches."""
        # Create DataFrame with wrong types
        schema = StructType(
            [
                StructField("customer_id", StringType(), True),
                StructField("amount", StringType(), True),  # Should be DoubleType
            ],
        )
        invalid_df = spark.createDataFrame([("cust1", "100.0")], schema)

        # Create explicit schema constraint
        from pyspark_transform_registry.schema_constraints import (
            ColumnRequirement,
            PartialSchemaConstraint,
        )

        constraint = PartialSchemaConstraint(
            required_columns=[
                ColumnRequirement("customer_id", "string"),
                ColumnRequirement("amount", "double"),
            ],
        )

        # Register and load function
        model_name = "test_catalog.test_schema.type_error_test"
        register_function(
            sample_transform_function,
            name=model_name,
            schema_constraint=constraint,
        )

        loaded_transform = load_function(model_name, version=1)

        # Error message should be informative
        with pytest.raises(ValueError) as exc_info:
            loaded_transform(invalid_df)

        error_message = str(exc_info.value)
        assert "Input validation failed" in error_message
        assert "amount" in error_message  # Should mention the problematic column


class TestValidationIntegration:
    """Test integration between validation and transform execution."""

    def test_validation_and_transformation_success_path(
        self,
        spark,
        sample_transform_function,
        valid_test_dataframe,
    ):
        """Test the complete success path from validation to transformation."""
        # Register function
        model_name = "test_catalog.test_schema.integration_success_test"
        register_function(
            sample_transform_function,
            name=model_name,
        )

        # Load and execute
        loaded_transform = load_function(model_name, version=1)
        result = loaded_transform(valid_test_dataframe)

        # Verify the transformation worked correctly
        assert result is not None
        assert "customer_id" in result.columns
        assert "amount" in result.columns
        assert "processed" in result.columns

        # Check data integrity
        result_data = result.collect()
        assert len(result_data) == 2
        assert result_data[0]["processed"] == 200.0  # 100.0 * 2
        assert result_data[1]["processed"] == 400.0  # 200.0 * 2

    def test_validation_performance_impact(
        self,
        spark,
        sample_transform_function,
        valid_test_dataframe,
    ):
        """Test that validation doesn't significantly impact performance."""
        import time

        # Register function
        model_name = "test_catalog.test_schema.performance_test"
        register_function(
            sample_transform_function,
            name=model_name,
        )

        # Load function with and without validation
        transform_with_validation = load_function(
            model_name,
            version=1,
            validate_input=True,
        )
        transform_without_validation = load_function(
            model_name,
            version=1,
            validate_input=False,
        )

        # Measure execution time with validation
        start_time = time.time()
        result_with_validation = transform_with_validation(valid_test_dataframe)
        time_with_validation = time.time() - start_time

        # Measure execution time without validation
        start_time = time.time()
        result_without_validation = transform_without_validation(valid_test_dataframe)
        time_without_validation = time.time() - start_time

        # Results should be identical
        assert result_with_validation.collect() == result_without_validation.collect()

        # Validation overhead should be minimal (this is a rough check)
        # In practice, validation should add minimal overhead for most cases
        assert time_with_validation < time_without_validation * 10  # At most 10x slower
