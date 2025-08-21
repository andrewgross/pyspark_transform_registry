"""Tests for multi-input function support in PySpark Transform Registry."""

import pytest
from pyspark.sql import DataFrame

from pyspark_transform_registry import load_function, register_function


@pytest.fixture
def sample_df(spark_session):
    """Create a sample DataFrame for testing."""
    data = [
        {"id": 1, "value": 10, "category": "A"},
        {"id": 2, "value": 20, "category": "B"},
        {"id": 3, "value": 5, "category": "A"},
        {"id": 4, "value": 15, "category": "B"},
    ]
    return spark_session.createDataFrame(data)


def test_single_parameter_function_registration_and_loading(sample_df, mlflow_tracking):
    """Test that single parameter functions work as before."""

    def simple_transform(df: DataFrame) -> DataFrame:
        """Simple transform that selects all columns."""
        return df.select("*")

    # Register the function
    register_function(
        simple_transform,
        name="test_registry.simple_transform",
    )

    # Load the function
    loaded_transform = load_function("test_registry.simple_transform", version=1)

    # Test single parameter usage
    result = loaded_transform(sample_df)
    assert result.count() == 4
    assert set(result.columns) == {"id", "value", "category"}


def test_multi_parameter_function_registration_and_loading(sample_df, mlflow_tracking):
    """Test that multi-parameter functions work with the new API."""

    def filter_transform(
        df: DataFrame,
        min_value: int = 0,
        category_filter: str = None,
    ) -> DataFrame:
        """Filter DataFrame by value and optionally by category."""
        filtered_df = df.filter(df.value >= min_value)
        if category_filter:
            filtered_df = filtered_df.filter(df.category == category_filter)
        return filtered_df

    # Register the function with example parameters
    register_function(
        filter_transform,
        name="test_registry.filter_transform",
    )

    # Load the function
    loaded_transform = load_function("test_registry.filter_transform", version=1)

    # Test single parameter usage (uses defaults)
    result = loaded_transform(sample_df)
    assert result.count() == 4  # No filtering with defaults

    # Test multi-parameter usage
    result = loaded_transform(sample_df, params={"min_value": 10})
    assert result.count() == 3  # Values >= 10: 10, 20, 15

    # Test multi-parameter usage with multiple params
    result = loaded_transform(
        sample_df,
        params={"min_value": 10, "category_filter": "A"},
    )
    assert result.count() == 1  # Only id=1 with value=10, category=A

    # Verify the result content
    result_list = result.collect()
    assert len(result_list) == 1
    assert result_list[0]["id"] == 1
    assert result_list[0]["value"] == 10
    assert result_list[0]["category"] == "A"


def test_function_with_required_parameters(sample_df, mlflow_tracking):
    """Test functions that have required (non-default) parameters."""

    def required_param_transform(df: DataFrame, multiplier: int) -> DataFrame:
        """Transform that requires a multiplier parameter."""
        return df.withColumn("value", df.value * multiplier)

    # Register the function with example parameters
    register_function(
        required_param_transform,
        name="test_registry.required_param_transform",
    )

    # Load the function
    loaded_transform = load_function(
        "test_registry.required_param_transform",
        version=1,
    )

    # Test that calling without required params fails appropriately
    with pytest.raises(TypeError):
        loaded_transform(sample_df)  # Missing required multiplier

    # Test with required parameter
    result = loaded_transform(sample_df, params={"multiplier": 3})
    assert result.count() == 4

    # Check that values were multiplied
    result_list = result.collect()
    original_values = [10, 20, 5, 15]
    result_values = sorted([row["value"] for row in result_list])
    expected_values = sorted([v * 3 for v in original_values])
    assert result_values == expected_values


def test_mixed_default_and_required_parameters(sample_df, mlflow_tracking):
    """Test functions with both default and required parameters."""

    def mixed_param_transform(
        df: DataFrame,
        required_col: str,
        optional_value: int = 100,
    ) -> DataFrame:
        """Transform with both required and optional parameters."""
        return df.withColumn(required_col, df.value + optional_value)

    # Register the function
    register_function(
        mixed_param_transform,
        name="test_registry.mixed_param_transform",
    )

    # Load the function
    loaded_transform = load_function("test_registry.mixed_param_transform", version=1)

    # Test with required parameter only (should use default for optional)
    result = loaded_transform(sample_df, params={"required_col": "calculated"})
    assert result.count() == 4
    assert "calculated" in result.columns

    # Verify default value was used (value + 100)
    result_list = result.collect()
    assert result_list[0]["calculated"] == 110  # 10 + 100

    # Test with both required and optional parameters
    result = loaded_transform(
        sample_df,
        params={"required_col": "custom", "optional_value": 5},
    )
    assert "custom" in result.columns

    # Verify custom optional value was used (value + 5)
    result_list = result.collect()
    assert result_list[0]["custom"] == 15  # 10 + 5


def test_backward_compatibility_with_existing_functions(sample_df, mlflow_tracking):
    """Test that existing single-parameter functions still work after the changes."""

    def legacy_transform(df: DataFrame) -> DataFrame:
        """Legacy transform function with single parameter."""
        return df.select("id", "value")

    # Register without any example params (old way)
    register_function(
        legacy_transform,
        name="test_registry.legacy_transform",
    )

    # Load and use the function
    loaded_transform = load_function("test_registry.legacy_transform", version=1)

    # Should work with single parameter
    result = loaded_transform(sample_df)
    assert result.count() == 4
    assert set(result.columns) == {"id", "value"}

    # Should also work with explicit None params (though not necessary)
    result = loaded_transform(sample_df, params=None)
    assert result.count() == 4
    assert set(result.columns) == {"id", "value"}
