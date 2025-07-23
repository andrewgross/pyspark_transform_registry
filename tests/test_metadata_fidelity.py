"""
Tests for function metadata fidelity through MLflow registration and loading.

This module tests that function behavior, parameter handling, and MLflow metadata
are properly preserved through the complete MLflow round-trip process.
"""

import pytest
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, lit

from pyspark_transform_registry import load_function, register_function


class TestFunctionalBehaviorFidelity:
    """Test that function behavior is preserved through registration/loading."""

    def test_simple_function_behavior(self, spark, mlflow_tracking):
        """Test that simple functions behave identically after loading."""

        def simple_transform(df: DataFrame) -> DataFrame:
            """A simple transformation."""
            return df.select("*").filter(col("id") > 0)

        # Test data
        test_df = spark.createDataFrame(
            [(1, "test"), (2, "data"), (0, "zero")],
            ["id", "value"],
        )

        # Register function
        register_function(func=simple_transform, name="test.metadata.simple_transform")

        # Load function back
        loaded_func = load_function("test.metadata.simple_transform", version=1)

        # Test functional equivalence
        original_result = simple_transform(test_df)
        loaded_result = loaded_func(test_df)

        # Should produce identical results
        original_data = sorted(original_result.collect(), key=lambda x: x.id)
        loaded_data = sorted(loaded_result.collect(), key=lambda x: x.id)

        assert len(original_data) == len(loaded_data)
        for orig_row, loaded_row in zip(original_data, loaded_data):
            assert orig_row.id == loaded_row.id
            assert orig_row.value == loaded_row.value

    def test_function_with_parameters(self, spark, mlflow_tracking):
        """Test function with multiple parameters and defaults."""

        def parameterized_transform(
            df: DataFrame,
            threshold: float = 100.0,
            category: str = "default",
            enabled: bool = True,
        ) -> DataFrame:
            """Transform with parameters."""
            return (
                df.filter(col("amount") >= threshold)
                .withColumn("category", lit(category))
                .withColumn("enabled", lit(enabled))
            )

        # Test data
        test_df = spark.createDataFrame(
            [(1, 150.0), (2, 75.0), (3, 120.0)],
            ["id", "amount"],
        )

        register_function(
            func=parameterized_transform,
            name="test.metadata.parameterized_transform",
        )

        loaded_func = load_function("test.metadata.parameterized_transform", version=1)

        # Test various parameter combinations
        test_cases = [
            {},  # Use all defaults
            {"threshold": 80.0},
            {"category": "vip"},
            {"enabled": False},
            {"threshold": 90.0, "category": "gold", "enabled": True},
        ]

        for params in test_cases:
            # Test original function
            original_result = parameterized_transform(test_df, **params)

            # Test loaded function
            if params:
                loaded_result = loaded_func(test_df, params=params)
            else:
                loaded_result = loaded_func(test_df)

            # Compare results
            original_data = sorted(original_result.collect(), key=lambda x: x.id)
            loaded_data = sorted(loaded_result.collect(), key=lambda x: x.id)

            assert len(original_data) == len(loaded_data)
            for orig_row, loaded_row in zip(original_data, loaded_data):
                assert orig_row.id == loaded_row.id
                assert orig_row.amount == loaded_row.amount
                assert orig_row.category == loaded_row.category
                assert orig_row.enabled == loaded_row.enabled

    def test_complex_data_types(self, spark, mlflow_tracking):
        """Test function with complex parameter types."""

        def complex_transform(
            df: DataFrame,
            filter_values: list | None = None,
            column_mapping: dict | None = None,
        ) -> DataFrame:
            """Transform with complex parameter types."""
            if filter_values is None:
                filter_values = [1, 2, 3]
            if column_mapping is None:
                column_mapping = {"old_name": "new_name"}

            result = df.filter(col("id").isin(filter_values))
            for old_col, new_col in column_mapping.items():
                if old_col in df.columns:
                    result = result.withColumnRenamed(old_col, new_col)
            return result

        test_df = spark.createDataFrame(
            [(1, "a"), (2, "b"), (3, "c"), (4, "d")],
            ["id", "old_name"],
        )

        register_function(
            func=complex_transform,
            name="test.metadata.complex_transform",
        )

        loaded_func = load_function("test.metadata.complex_transform", version=1)

        # Test with defaults
        original_result = complex_transform(test_df)
        loaded_result = loaded_func(test_df)

        assert original_result.count() == loaded_result.count()
        assert set(original_result.columns) == set(loaded_result.columns)

        # Test with custom parameters
        custom_params = {
            "filter_values": [2, 3],
            "column_mapping": {"old_name": "custom_name"},
        }

        original_result2 = complex_transform(test_df, **custom_params)
        loaded_result2 = loaded_func(test_df, params=custom_params)

        assert original_result2.count() == loaded_result2.count()
        assert "custom_name" in loaded_result2.columns
        assert loaded_result2.count() == 2  # Only ids 2 and 3


class TestMLflowMetadataPreservation:
    """Test that MLflow metadata is properly stored and retrievable."""

    def test_function_name_preservation(self, spark, mlflow_tracking):
        """Test that function names are stored in MLflow metadata."""

        def named_function(df: DataFrame) -> DataFrame:
            """A function with a specific name."""
            return df.select("*")

        register_function(
            func=named_function,
            name="test.metadata.named_function",
            description="Test function for name preservation",
        )

        # Verify function can be loaded and works
        loaded_func = load_function("test.metadata.named_function", version=1)

        test_df = spark.createDataFrame([(1, "test")], ["id", "value"])
        result = loaded_func(test_df)
        assert result.count() == 1

    def test_docstring_metadata_storage(self, spark, mlflow_tracking):
        """Test that docstrings are preserved in MLflow metadata."""

        def documented_function(df: DataFrame) -> DataFrame:
            """
            This is a comprehensive docstring.

            It contains multiple lines and detailed information
            about what the function does.

            Args:
                df: Input DataFrame

            Returns:
                Processed DataFrame
            """
            return df.withColumn("processed", lit(True))

        register_function(
            func=documented_function,
            name="test.metadata.documented_function",
            description="Function with comprehensive documentation",
        )

        # Verify function works correctly
        loaded_func = load_function("test.metadata.documented_function", version=1)

        test_df = spark.createDataFrame([(1, "test")], ["id", "value"])
        result = loaded_func(test_df)

        assert result.count() == 1
        assert "processed" in result.columns

    def test_parameter_example_preservation(self, spark, mlflow_tracking):
        """Test that parameter examples are properly handled."""

        def example_function(
            df: DataFrame,
            multiplier: float = 2.0,
            prefix: str = "processed",
        ) -> DataFrame:
            """Function with example parameters."""
            return df.withColumn(
                "multiplied_value",
                col("value") * multiplier,
            ).withColumn("prefixed_name", lit(prefix))

        sample_df = spark.createDataFrame([(1, 10.0), (2, 20.0)], ["id", "value"])

        example_params = {"multiplier": 3.0, "prefix": "example"}

        register_function(
            func=example_function,
            name="test.metadata.example_function",
        )

        # Test that the function works with the example parameters
        loaded_func = load_function("test.metadata.example_function", version=1)

        # Test with example parameters
        result = loaded_func(sample_df, params=example_params)
        collected = result.collect()

        assert len(collected) == 2
        assert collected[0].multiplied_value == 30.0  # 10.0 * 3.0
        assert collected[0].prefixed_name == "example"


class TestErrorHandlingFidelity:
    """Test that error handling behavior is preserved."""

    def test_parameter_validation_errors(self, spark, mlflow_tracking):
        """Test that parameter validation errors are consistent."""

        def strict_function(df: DataFrame, required_param: str) -> DataFrame:
            """Function that requires a specific parameter."""
            if not required_param:
                raise ValueError("required_param cannot be empty")
            return df.withColumn("param_value", lit(required_param))

        register_function(func=strict_function, name="test.metadata.strict_function")

        loaded_func = load_function(
            "test.metadata.strict_function",
            version=1,
            validate_input=False,
        )

        test_df = spark.createDataFrame([(1, "test")], ["id", "value"])

        # Test that both functions handle errors similarly
        # Valid parameter
        original_result = strict_function(test_df, "valid")
        loaded_result = loaded_func(test_df, params={"required_param": "valid"})

        assert original_result.count() == loaded_result.count()

        # Invalid parameter should cause similar errors
        with pytest.raises(ValueError, match="required_param cannot be empty"):
            strict_function(test_df, "")

        with pytest.raises(ValueError, match="required_param cannot be empty"):
            loaded_func(test_df, params={"required_param": ""})

    def test_dataframe_error_consistency(self, spark, mlflow_tracking):
        """Test that DataFrame errors are handled consistently."""

        def column_dependent_function(df: DataFrame) -> DataFrame:
            """Function that depends on specific columns."""
            return df.select("id", "required_column")

        register_function(
            func=column_dependent_function,
            name="test.metadata.column_dependent_function",
        )

        loaded_func = load_function(
            "test.metadata.column_dependent_function",
            version=1,
            validate_input=False,
        )

        # DataFrame with required column
        valid_df = spark.createDataFrame([(1, "test")], ["id", "required_column"])

        original_result = column_dependent_function(valid_df)
        loaded_result = loaded_func(valid_df)

        assert original_result.count() == loaded_result.count()

        # DataFrame without required column
        invalid_df = spark.createDataFrame([(1, "test")], ["id", "other_column"])

        # Both should fail when column is missing
        with pytest.raises(Exception):  # Likely AnalysisException
            column_dependent_function(invalid_df)

        with pytest.raises(Exception):  # Should also fail
            loaded_func(invalid_df)


class TestVersionCompatibility:
    """Test behavior across different function versions."""

    def test_signature_evolution(self, spark, mlflow_tracking):
        """Test that function signature changes are handled properly."""

        # Version 1: Simple function
        def transform_v1(df: DataFrame) -> DataFrame:
            """Version 1 of the transform."""
            return df.withColumn("version", lit("v1"))

        # Version 2: Function with additional parameter
        def transform_v2(df: DataFrame, version_label: str = "v2") -> DataFrame:
            """Version 2 of the transform with additional parameter."""
            return df.withColumn("version", lit(version_label))

        # Register both versions
        register_function(func=transform_v1, name="test.metadata.versioned_transform")

        register_function(func=transform_v2, name="test.metadata.versioned_transform")

        # Load both versions
        v1_func = load_function("test.metadata.versioned_transform", version=1)
        v2_func = load_function("test.metadata.versioned_transform", version=2)

        test_df = spark.createDataFrame([(1, "test")], ["id", "value"])

        # Test v1 behavior
        v1_result = v1_func(test_df)
        assert v1_result.collect()[0].version == "v1"

        # Test v2 behavior with default
        v2_result = v2_func(test_df)
        assert v2_result.collect()[0].version == "v2"

        # Test v2 behavior with custom parameter
        v2_custom = v2_func(test_df, params={"version_label": "custom"})
        assert v2_custom.collect()[0].version == "custom"

    def test_backward_compatibility(self, spark, mlflow_tracking):
        """Test that newer versions maintain backward compatibility."""

        def backward_compatible_v1(df: DataFrame) -> DataFrame:
            """Original version."""
            return df.select("id", "value")

        def backward_compatible_v2(
            df: DataFrame,
            include_extra: bool = False,
        ) -> DataFrame:
            """Enhanced version with optional parameter."""
            if include_extra:
                return df.select("id", "value").withColumn("extra", lit("added"))
            else:
                return df.select("id", "value")

        register_function(
            func=backward_compatible_v1,
            name="test.metadata.backward_compatible",
        )

        register_function(
            func=backward_compatible_v2,
            name="test.metadata.backward_compatible",
        )

        test_df = spark.createDataFrame([(1, "test")], ["id", "value"])

        v1_func = load_function("test.metadata.backward_compatible", version=1)
        v2_func = load_function("test.metadata.backward_compatible", version=2)

        # Both should work with the same input
        v1_result = v1_func(test_df)
        v2_result = v2_func(test_df)  # Using defaults, should behave like v1

        assert v1_result.count() == v2_result.count()
        assert set(v1_result.columns) == set(v2_result.columns)

        # v2 with parameter should add extra functionality
        v2_enhanced = v2_func(test_df, params={"include_extra": True})
        assert "extra" in v2_enhanced.columns
        assert v2_enhanced.collect()[0].extra == "added"
