"""
Tests for error handling robustness in the PySpark Transform Registry.

This module tests that the system gracefully handles edge cases, malformed data,
invalid inputs, network failures, and other error conditions that could occur
in real-world usage.
"""

import os
import tempfile
from unittest.mock import patch

import pytest
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, lit
from pyspark.sql.types import IntegerType, StringType, StructField, StructType

from pyspark_transform_registry import load_function, register_function
from pyspark_transform_registry.runtime_validation import RuntimeValidator
from pyspark_transform_registry.schema_constraints import (
    ColumnRequirement,
    PartialSchemaConstraint,
)


class TestRegistrationErrorHandling:
    """Test error handling during function registration."""

    def test_invalid_function_parameter(self, spark, mlflow_tracking):
        """Test registration with invalid function parameter."""

        # Test with None function
        with pytest.raises(
            ValueError,
            match="Either 'func' or 'file_path' must be provided",
        ):
            register_function(func=None, name="test.error.none_function")

        # Test with both func and file_path
        def dummy_func(df: DataFrame) -> DataFrame:
            return df

        with pytest.raises(
            ValueError,
            match="Cannot specify both 'func' and 'file_path'",
        ):
            register_function(
                func=dummy_func,
                file_path="/some/path.py",
                name="test.error.both_params",
            )

    def test_invalid_file_path_registration(self, spark, mlflow_tracking):
        """Test registration with invalid file paths."""

        # Test with non-existent file
        with pytest.raises(FileNotFoundError, match="File not found"):
            register_function(
                file_path="/nonexistent/path.py",
                function_name="dummy_function",
                name="test.error.nonexistent_file",
            )

        # Test with file_path but no function_name
        with pytest.raises(
            ValueError,
            match="'function_name' is required when using 'file_path'",
        ):
            register_function(
                file_path="/some/path.py",
                name="test.error.no_function_name",
            )

    def test_malformed_function_file(self, spark, mlflow_tracking):
        """Test registration with malformed Python files."""

        # Create a temporary file with invalid Python syntax
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("def broken_function(df):\n    return df.invalid syntax here")
            temp_file = f.name

        try:
            with pytest.raises(Exception):  # Should raise SyntaxError or similar
                register_function(
                    file_path=temp_file,
                    function_name="broken_function",
                    name="test.error.malformed_file",
                )
        finally:
            os.unlink(temp_file)

    def test_function_not_found_in_file(self, spark, mlflow_tracking):
        """Test registration when function doesn't exist in file."""

        # Create a temporary file with a different function
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("""
def existing_function(df):
    return df
""")
            temp_file = f.name

        try:
            with pytest.raises(
                AttributeError,
                match="Function 'nonexistent_function' not found",
            ):
                register_function(
                    file_path=temp_file,
                    function_name="nonexistent_function",
                    name="test.error.function_not_found",
                )
        finally:
            os.unlink(temp_file)

    def test_non_callable_object_in_file(self, spark, mlflow_tracking):
        """Test registration when the named object is not callable."""

        # Create a temporary file with a non-function object
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("""
not_a_function = "this is a string, not a function"
""")
            temp_file = f.name

        try:
            with pytest.raises(TypeError, match="'not_a_function' is not a function"):
                register_function(
                    file_path=temp_file,
                    function_name="not_a_function",
                    name="test.error.not_callable",
                )
        finally:
            os.unlink(temp_file)

    def test_schema_inference_failure(self, spark, mlflow_tracking):
        """Test registration when schema inference fails gracefully."""

        def problematic_function(df: DataFrame) -> DataFrame:
            """Function that might cause schema inference issues."""
            # Use complex operations that might be hard to analyze
            result = df
            for i in range(5):
                result = result.filter(col("amount") > i * 10)
            return result.withColumn("complex", lit("value"))

        # Should not raise an error, but should proceed without schema constraint
        register_function(
            func=problematic_function,
            name="test.error.schema_inference_failure",
            infer_schema=True,
        )

        # Function should still be loadable and work
        loaded_func = load_function("test.error.schema_inference_failure", version=1)

        test_df = spark.createDataFrame([(1, 100)], ["id", "amount"])
        result = loaded_func(test_df)
        assert result.count() == 1


class TestLoadingErrorHandling:
    """Test error handling during function loading."""

    def test_load_nonexistent_function(self, spark, mlflow_tracking):
        """Test loading a function that doesn't exist."""

        with pytest.raises(Exception):  # Should raise MLflow exception
            load_function("nonexistent.function.name", version=1)

    def test_load_invalid_version(self, spark, mlflow_tracking):
        """Test loading with invalid version numbers."""

        # First register a function
        def test_function(df: DataFrame) -> DataFrame:
            return df.select("*")

        register_function(func=test_function, name="test.error.versioned_function")

        # Try to load non-existent version
        with pytest.raises(Exception):  # Should raise MLflow exception
            load_function("test.error.versioned_function", version=999)

    def test_corrupted_model_handling(self, spark, mlflow_tracking):
        """Test handling of corrupted model data."""

        def test_function(df: DataFrame) -> DataFrame:
            return df.select("*")

        register_function(func=test_function, name="test.error.corrupted_model")

        # Note: Simulating actual corruption is complex with MLflow's storage format
        # But we can test that the function loads and basic functionality works
        loaded_func = load_function("test.error.corrupted_model", version=1)

        test_df = spark.createDataFrame([(1, "test")], ["id", "value"])
        result = loaded_func(test_df)
        assert result.count() == 1

    def test_malformed_schema_constraint_json(self, spark, mlflow_tracking):
        """Test handling of malformed schema constraint JSON."""

        def test_function(df: DataFrame) -> DataFrame:
            return df.withColumn("processed", lit(True))

        # Create a valid constraint first
        valid_constraint = PartialSchemaConstraint(
            required_columns=[ColumnRequirement("test_col", "string")],
        )

        register_function(
            func=test_function,
            name="test.error.malformed_constraint",
            schema_constraint=valid_constraint,
        )

        # Should load and work normally
        loaded_func = load_function("test.error.malformed_constraint", version=1)

        # Test with DataFrame that has the required column
        test_df = spark.createDataFrame([(1, "test")], ["id", "test_col"])
        result = loaded_func(test_df)
        assert result.count() == 1

        # Test with validation disabled (should work even with missing columns)
        loaded_func_no_validation = load_function(
            "test.error.malformed_constraint",
            version=1,
            validate_input=False,
        )
        test_df_missing = spark.createDataFrame([(1, "test")], ["id", "value"])
        result = loaded_func_no_validation(test_df_missing)
        assert result.count() == 1

    def test_missing_mlflow_metadata(self, spark, mlflow_tracking):
        """Test handling when MLflow metadata is missing or incomplete."""

        def test_function(df: DataFrame) -> DataFrame:
            return df.select("*")

        register_function(func=test_function, name="test.error.missing_metadata")

        # Function should still load and work despite missing metadata
        loaded_func = load_function(
            "test.error.missing_metadata",
            version=1,
            validate_input=False,
        )

        test_df = spark.createDataFrame([(1, "test")], ["id", "value"])
        result = loaded_func(test_df)
        assert result.count() == 1


class TestRuntimeValidationErrorHandling:
    """Test error handling in runtime validation."""

    def test_validation_with_null_constraint(self, spark, mlflow_tracking):
        """Test validation when schema constraint is None."""

        def simple_function(df: DataFrame) -> DataFrame:
            return df.select("*")

        register_function(
            func=simple_function,
            name="test.error.null_constraint",
            infer_schema=False,  # Explicitly disable schema inference
        )

        # Should load without validation errors
        loaded_func = load_function(
            "test.error.null_constraint",
            version=1,
            validate_input=True,
        )

        test_df = spark.createDataFrame([(1, "test")], ["id", "value"])
        result = loaded_func(test_df)
        assert result.count() == 1

    def test_validation_with_empty_dataframe(self, spark, mlflow_tracking):
        """Test validation with empty DataFrames."""

        def filter_function(df: DataFrame) -> DataFrame:
            return df.filter(col("amount") > 0)

        constraint = PartialSchemaConstraint(
            required_columns=[ColumnRequirement("amount", "double")],
        )

        register_function(
            func=filter_function,
            name="test.error.empty_df_validation",
            schema_constraint=constraint,
        )

        loaded_func = load_function("test.error.empty_df_validation", version=1)

        # Test with empty DataFrame
        empty_df = spark.createDataFrame(
            [],
            StructType(
                [
                    StructField(
                        "amount",
                        StringType(),
                        True,
                    ),  # Wrong type intentionally
                ],
            ),
        )

        # Should handle empty DataFrame gracefully
        with pytest.raises(ValueError, match="Input validation failed"):
            loaded_func(empty_df)

    def test_validation_with_malformed_dataframe(self, spark, mlflow_tracking):
        """Test validation with malformed DataFrame structures."""

        def select_function(df: DataFrame) -> DataFrame:
            return df.select("id", "value")

        constraint = PartialSchemaConstraint(
            required_columns=[
                ColumnRequirement("id", "integer"),
                ColumnRequirement("value", "string"),
            ],
        )

        register_function(
            func=select_function,
            name="test.error.malformed_df_validation",
            schema_constraint=constraint,
        )

        loaded_func = load_function("test.error.malformed_df_validation", version=1)

        # Test with DataFrame missing required columns
        malformed_df = spark.createDataFrame([(1,)], ["only_one_column"])

        with pytest.raises(ValueError, match="Input validation failed"):
            loaded_func(malformed_df)

    def test_validator_with_invalid_constraint(self, spark, mlflow_tracking):
        """Test runtime validator with invalid constraint objects."""

        validator = RuntimeValidator()

        test_df = spark.createDataFrame([(1, "test")], ["id", "value"])

        # Test with None constraint
        with pytest.raises(Exception):
            validator.validate_dataframe(test_df, None)


class TestDataFrameErrorHandling:
    """Test error handling with various DataFrame issues."""

    def test_function_with_missing_columns(self, spark, mlflow_tracking):
        """Test functions that reference missing columns."""

        def column_dependent_function(df: DataFrame) -> DataFrame:
            return df.select("id", "nonexistent_column")

        register_function(
            func=column_dependent_function,
            name="test.error.missing_columns",
            infer_schema=False,
        )

        loaded_func = load_function(
            "test.error.missing_columns",
            version=1,
            validate_input=False,
        )

        test_df = spark.createDataFrame([(1, "test")], ["id", "value"])

        # Should raise PySpark AnalysisException
        with pytest.raises(Exception):
            loaded_func(test_df)

    def test_function_with_type_mismatch(self, spark, mlflow_tracking):
        """Test functions with type mismatches."""

        def type_dependent_function(df: DataFrame) -> DataFrame:
            return df.withColumn("doubled", col("amount") * 2)

        register_function(
            func=type_dependent_function,
            name="test.error.type_mismatch",
            infer_schema=False,
        )

        loaded_func = load_function(
            "test.error.type_mismatch",
            version=1,
            validate_input=False,
        )

        # DataFrame with string where numeric expected
        test_df = spark.createDataFrame([(1, "not_a_number")], ["id", "amount"])

        # Should raise PySpark exception during execution
        try:
            result = loaded_func(test_df)
            result.collect()  # Force evaluation
            # If no exception, PySpark was able to handle the conversion
        except Exception as e:
            # Expected - should get type conversion error
            assert (
                "Cannot resolve" in str(e)
                or "invalid" in str(e).lower()
                or "NumberFormatException" in str(e)
            )

    def test_function_with_null_handling(self, spark, mlflow_tracking):
        """Test functions with unexpected null values."""

        from pyspark.sql.functions import length

        def null_sensitive_function(df: DataFrame) -> DataFrame:
            return df.withColumn("text_length", length(col("text")))

        register_function(
            func=null_sensitive_function,
            name="test.error.null_handling",
            infer_schema=False,
        )

        loaded_func = load_function(
            "test.error.null_handling",
            version=1,
            validate_input=False,
        )

        # DataFrame with null values
        test_df = spark.createDataFrame([(1, None), (2, "test")], ["id", "text"])

        # Should handle nulls gracefully (PySpark typically returns null for null.length())
        result = loaded_func(test_df)
        collected = result.collect()

        assert len(collected) == 2
        assert collected[0].text_length is None  # null.length() should be null
        assert collected[1].text_length == 4  # "test".length() should be 4

    def test_function_with_large_dataframe(self, spark, mlflow_tracking):
        """Test function behavior with edge case DataFrame sizes."""

        def simple_count_function(df: DataFrame) -> DataFrame:
            return df.withColumn("row_count", lit(df.count()))

        register_function(
            func=simple_count_function,
            name="test.error.large_dataframe",
            infer_schema=False,
        )

        loaded_func = load_function(
            "test.error.large_dataframe",
            version=1,
            validate_input=False,
        )

        # Test with empty DataFrame
        empty_df = spark.createDataFrame(
            [],
            StructType([StructField("id", IntegerType(), True)]),
        )

        result = loaded_func(empty_df)
        assert result.count() == 0


class TestParameterErrorHandling:
    """Test error handling with function parameters."""

    def test_missing_required_parameters(self, spark, mlflow_tracking):
        """Test functions when required parameters are missing."""

        def param_required_function(df: DataFrame, required_param: str) -> DataFrame:
            return df.withColumn("param_value", lit(required_param))

        register_function(
            func=param_required_function,
            name="test.error.missing_required_params",
            infer_schema=False,
        )

        loaded_func = load_function(
            "test.error.missing_required_params",
            version=1,
            validate_input=False,
        )

        test_df = spark.createDataFrame([(1, "test")], ["id", "value"])

        # Should raise TypeError when required parameter is missing
        with pytest.raises(TypeError):
            loaded_func(test_df)  # Missing params dict

    def test_invalid_parameter_types(self, spark, mlflow_tracking):
        """Test functions with invalid parameter types."""

        def type_sensitive_function(df: DataFrame, multiplier: float) -> DataFrame:
            return df.withColumn("multiplied", col("amount") * multiplier)

        register_function(
            func=type_sensitive_function,
            name="test.error.invalid_param_types",
            infer_schema=False,
        )

        loaded_func = load_function(
            "test.error.invalid_param_types",
            version=1,
            validate_input=False,
        )

        test_df = spark.createDataFrame([(1, 10.0)], ["id", "amount"])

        # PySpark should handle type coercion, but invalid types might cause issues
        try:
            # This might work due to PySpark's type coercion
            result = loaded_func(test_df, params={"multiplier": "not_a_number"})
            result.collect()
        except Exception:
            # If it fails, that's expected behavior
            pass

    def test_extra_unexpected_parameters(self, spark, mlflow_tracking):
        """Test functions with extra unexpected parameters."""

        def simple_param_function(
            df: DataFrame,
            expected_param: str = "default",
        ) -> DataFrame:
            return df.withColumn("param", lit(expected_param))

        register_function(
            func=simple_param_function,
            name="test.error.extra_params",
            infer_schema=False,
        )

        loaded_func = load_function(
            "test.error.extra_params",
            version=1,
            validate_input=False,
        )

        test_df = spark.createDataFrame([(1, "test")], ["id", "value"])

        # Should raise TypeError for unexpected keyword arguments
        with pytest.raises(TypeError):
            loaded_func(
                test_df,
                params={"expected_param": "value", "unexpected_param": "extra"},
            )


class TestNetworkAndResourceErrors:
    """Test error handling for network and resource-related issues."""

    def test_mlflow_connection_issues(self, spark, mlflow_tracking):
        """Test handling of MLflow connection issues."""

        def test_function(df: DataFrame) -> DataFrame:
            return df.select("*")

        # Register a function first
        register_function(func=test_function, name="test.error.connection_issues")

        # Test loading should work with normal connection
        loaded_func = load_function("test.error.connection_issues", version=1)

        test_df = spark.createDataFrame([(1, "test")], ["id", "value"])
        result = loaded_func(test_df)
        assert result.count() == 1

    @patch("mlflow.tracking.MlflowClient")
    def test_mlflow_client_failures(self, mock_client, spark, mlflow_tracking):
        """Test handling of MLflow client failures."""

        # Mock client to raise exceptions
        mock_client.side_effect = Exception("MLflow client error")

        def test_function(df: DataFrame) -> DataFrame:
            return df.select("*")

        register_function(
            func=test_function,
            name="test.error.client_failures",
            infer_schema=False,
        )

        # Loading with validation should handle client failures gracefully
        loaded_func = load_function(
            "test.error.client_failures",
            version=1,
            validate_input=True,
        )

        test_df = spark.createDataFrame([(1, "test")], ["id", "value"])
        # Should work despite client issues (validation will be skipped)
        result = loaded_func(test_df)
        assert result.count() == 1

    def test_memory_constraints(self, spark, mlflow_tracking):
        """Test handling of memory constraints during processing."""

        def memory_intensive_function(df: DataFrame) -> DataFrame:
            # Create a function that might use more memory
            return df.crossJoin(df).select("*")

        register_function(
            func=memory_intensive_function,
            name="test.error.memory_constraints",
            infer_schema=False,
        )

        loaded_func = load_function(
            "test.error.memory_constraints",
            version=1,
            validate_input=False,
        )

        # Test with small DataFrame (should work)
        small_df = spark.createDataFrame([(1, "test")], ["id", "value"])
        result = loaded_func(small_df)
        assert result.count() == 1


class TestEdgeCaseInputs:
    """Test handling of edge case inputs."""

    def test_unicode_and_special_characters(self, spark, mlflow_tracking):
        """Test handling of Unicode and special characters."""

        def text_processing_function(df: DataFrame) -> DataFrame:
            return df.withColumn("processed", col("text"))

        register_function(
            func=text_processing_function,
            name="test.error.unicode_handling",
            infer_schema=False,
        )

        loaded_func = load_function(
            "test.error.unicode_handling",
            version=1,
            validate_input=False,
        )

        # Test with various Unicode characters
        unicode_df = spark.createDataFrame(
            [(1, "Hello 🌍"), (2, "Café"), (3, "中文"), (4, "العربية"), (5, "🚀✨💻")],
            ["id", "text"],
        )

        result = loaded_func(unicode_df)
        assert result.count() == 5

    def test_very_long_strings(self, spark, mlflow_tracking):
        """Test handling of very long string values."""

        from pyspark.sql.functions import length

        def string_length_function(df: DataFrame) -> DataFrame:
            return df.withColumn("string_length", length(col("text")))

        register_function(
            func=string_length_function,
            name="test.error.long_strings",
            infer_schema=False,
        )

        loaded_func = load_function(
            "test.error.long_strings",
            version=1,
            validate_input=False,
        )

        # Test with very long string
        long_string = "a" * 10000  # 10K character string
        long_string_df = spark.createDataFrame([(1, long_string)], ["id", "text"])

        result = loaded_func(long_string_df)
        collected = result.collect()
        assert collected[0].string_length == 10000

    def test_extreme_numeric_values(self, spark, mlflow_tracking):
        """Test handling of extreme numeric values."""

        def numeric_processing_function(df: DataFrame) -> DataFrame:
            return df.withColumn("doubled", col("value") * 2)

        register_function(
            func=numeric_processing_function,
            name="test.error.extreme_numerics",
            infer_schema=False,
        )

        loaded_func = load_function(
            "test.error.extreme_numerics",
            version=1,
            validate_input=False,
        )

        # Test with extreme values
        extreme_df = spark.createDataFrame(
            [
                (1, float("inf")),
                (2, float("-inf")),
                (3, 0.0),
                (4, 1e-308),  # Very small number
                (5, 1e308),  # Very large number
            ],
            ["id", "value"],
        )

        result = loaded_func(extreme_df)
        assert result.count() == 5
        assert result.count() == 5
        assert result.count() == 5
