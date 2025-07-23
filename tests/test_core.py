"""
Unit tests for the new simplified core module.
"""

import pytest
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, lit

from pyspark_transform_registry.core import (
    _load_function_from_file,
    load_function,
    register_function,
)


class TestDirectRegistration:
    """Test direct function registration."""

    def test_register_simple_function(self, spark, mlflow_tracking):
        """Test registering a simple function directly."""

        def simple_transform(df: DataFrame) -> DataFrame:
            return df.select("*")

        # Register function
        logged_model = register_function(
            func=simple_transform,
            name="test.simple.transform",
            description="A simple transform function",
        )
        assert logged_model is not None
        assert logged_model.name == "simple_transform"
        assert logged_model.registered_model_version == 1

    def test_register_function_with_parameters(self, spark, mlflow_tracking):
        """Test registering a function with parameters."""

        def filter_transform(df: DataFrame, min_value: int = 0) -> DataFrame:
            return df.filter(col("value") > min_value)

        logged_model = register_function(
            func=filter_transform,
            name="test.filter.transform",
            extra_pip_requirements=["pandas>=1.0.0"],
            tags={"category": "filter", "author": "test"},
        )

        assert logged_model is not None
        assert logged_model.name == "filter_transform"
        assert logged_model.tags["category"] == "filter"
        assert logged_model.tags["author"] == "test"

    def test_register_function_missing_args(self):
        """Test that registration fails with missing arguments."""

        with pytest.raises(
            ValueError,
            match="Either 'func' or 'file_path' must be provided",
        ):
            register_function(name="test.missing.args")

    def test_register_function_conflicting_args(self, spark):
        """Test that registration fails with conflicting arguments."""

        def dummy_func(df: DataFrame) -> DataFrame:
            return df

        with pytest.raises(
            ValueError,
            match="Cannot specify both 'func' and 'file_path'",
        ):
            register_function(
                func=dummy_func,
                file_path="some_file.py",
                name="test.conflict.args",
            )


class TestFileBasedRegistration:
    """Test file-based function registration."""

    def test_register_from_file(self, spark, mlflow_tracking):
        """Test registering a function from a file."""

        # Register function from fixture file
        logged_model = register_function(
            file_path="tests/fixtures/simple_transform.py",
            function_name="simple_filter",
            name="test.file.simple_filter",
        )

        assert logged_model is not None
        assert logged_model.name == "simple_filter"
        assert logged_model.registered_model_version == 1

    def test_register_from_file_missing_function_name(self):
        """Test that file-based registration fails without function name."""

        with pytest.raises(
            ValueError,
            match="'function_name' is required when using 'file_path'",
        ):
            register_function(
                file_path="tests/fixtures/simple_transform.py",
                name="test.file.missing_name",
            )

    def test_load_function_from_file(self):
        """Test loading a function from a file."""

        func = _load_function_from_file(
            "tests/fixtures/simple_transform.py",
            "simple_filter",
        )

        assert callable(func)
        assert func.__name__ == "simple_filter"

    def test_load_function_from_nonexistent_file(self):
        """Test loading from a non-existent file."""

        with pytest.raises(FileNotFoundError):
            _load_function_from_file("nonexistent.py", "some_function")

    def test_load_nonexistent_function_from_file(self):
        """Test loading a non-existent function from a file."""

        with pytest.raises(AttributeError, match="Function 'nonexistent' not found"):
            _load_function_from_file(
                "tests/fixtures/simple_transform.py",
                "nonexistent",
            )


class TestFunctionLoading:
    """Test function loading from registry."""

    def test_load_registered_function(self, spark, mlflow_tracking):
        """Test loading a registered function."""

        def test_transform(df: DataFrame) -> DataFrame:
            return df.withColumn("test_col", lit("loaded"))

        test_data = spark.createDataFrame([(1, "a")], ["id", "value"])

        # Register function
        register_function(
            func=test_transform,
            name="test.load.transform",
        )

        # Load function
        loaded_func = load_function("test.load.transform", version=1)

        # Test loaded function
        result = loaded_func(test_data)
        assert result.count() == 1
        assert "test_col" in result.columns

    def test_load_function_with_version(self, spark, mlflow_tracking):
        """Test loading a specific version of a function."""

        def test_transform_v1(df: DataFrame) -> DataFrame:
            return df.withColumn("version", lit("v1"))

        def test_transform_v2(df: DataFrame) -> DataFrame:
            return df.withColumn("version", lit("v2"))

        test_data = spark.createDataFrame([(1, "a")], ["id", "value"])

        # Register v1
        register_function(
            func=test_transform_v1,
            name="test.version.transform",
        )

        # Register v2
        register_function(
            func=test_transform_v2,
            name="test.version.transform",
        )

        # Load specific version
        loaded_func_v1 = load_function("test.version.transform", version=1)
        loaded_func_v2 = load_function("test.version.transform", version=2)

        # Test that different versions work
        result_v1 = loaded_func_v1(test_data)
        result_v2 = loaded_func_v2(test_data)

        # Both should work but may have different behavior
        assert result_v1.count() == 1
        assert result_v2.count() == 1


# Note: TestModelListing class removed - list_registered_functions() functionality eliminated
