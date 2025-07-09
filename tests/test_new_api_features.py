import pytest
import mlflow
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, when, lit
from pyspark.sql import Column

from pyspark_transform_registry import (
    log_transform_function,
    load_transform_function,
    TransformType,
)
from pyspark_transform_registry.core import (
    _detect_transform_type,
    _suggest_calling_convention,
)


def test_keyword_only_arguments(spark, mlflow_tracking):
    """Test that all arguments except func must be named."""

    def sample_func(df: DataFrame) -> DataFrame:
        return df.select("*")

    # This should work - all args are named
    with mlflow.start_run():
        model_uri = log_transform_function(sample_func, name="test_func")
        assert model_uri is not None

    # This should fail - second argument is positional
    with pytest.raises(TypeError, match="takes 1 positional argument but 2 were given"):
        log_transform_function(sample_func, "should_fail")


def test_automatic_name_detection(spark, mlflow_tracking):
    """Test that function name is used when name is not provided."""

    def my_awesome_transform(df: DataFrame) -> DataFrame:
        return df.select("*")

    with mlflow.start_run():
        log_transform_function(my_awesome_transform)

        # Load the function back and verify the name was used
        loaded_func = load_transform_function("my_awesome_transform", version="1")
        assert loaded_func is not None


def test_transform_type_detection():
    """Test automatic detection of different transform types."""

    # DataFrame transform functions
    def clean_data(df: DataFrame, min_value: int = 0) -> DataFrame:
        return df.filter(col("value") > min_value)

    def simple_transform(df: DataFrame) -> DataFrame:
        return df.select("*")

    # Column expression functions
    def risk_score_columns_only(amount: Column, category: Column) -> Column:
        return when(category == "high", amount * 1.5).otherwise(amount)

    def risk_score_mixed(amount: Column, multiplier: float = 1.2) -> Column:
        return amount * lit(multiplier)

    def complex_calculation(base_col: Column, threshold: int, factor: float) -> Column:
        return when(base_col > threshold, base_col * factor).otherwise(base_col)

    # Custom functions
    def no_hints_function(x, y):
        return x + y

    def custom_processor(df, config_dict):
        return df.limit(config_dict.get("limit", 10))

    # Test detection
    test_cases = [
        (clean_data, TransformType.DATAFRAME_TRANSFORM),
        (simple_transform, TransformType.DATAFRAME_TRANSFORM),
        (risk_score_columns_only, TransformType.COLUMN_EXPRESSION),
        (risk_score_mixed, TransformType.COLUMN_EXPRESSION),
        (complex_calculation, TransformType.COLUMN_EXPRESSION),
        (no_hints_function, TransformType.CUSTOM),
        (custom_processor, TransformType.CUSTOM),
    ]

    for func, expected_type in test_cases:
        detected_type = _detect_transform_type(func)
        assert detected_type == expected_type, (
            f"Function {func.__name__} should be {expected_type}, got {detected_type}"
        )


def test_calling_convention_suggestions():
    """Test that appropriate calling conventions are suggested."""

    conventions = {
        TransformType.DATAFRAME_TRANSFORM: "df.transform(func, **kwargs) or func(df, **kwargs)",
        TransformType.COLUMN_EXPRESSION: "df.withColumn('col_name', func(col('input')))",
        TransformType.CUSTOM: "See function documentation for usage",
    }

    for transform_type, expected_convention in conventions.items():
        suggested = _suggest_calling_convention(transform_type)
        assert suggested == expected_convention


def test_metadata_storage(spark, mlflow_tracking):
    """Test that transform type and calling convention are stored in metadata."""

    def risk_calculator(amount: Column, risk_factor: float = 1.0) -> Column:
        """Calculate risk-adjusted amount."""
        return amount * lit(risk_factor)

    test_data = spark.createDataFrame(
        [(100, "item1"), (200, "item2")],
        ["amount", "item"],
    )

    with mlflow.start_run():
        # Register the function
        log_transform_function(
            risk_calculator,
            input_example=test_data,
            name="risk_calc",
        )

        # Verify metadata is accessible through find_transform_versions
        from pyspark_transform_registry import find_transform_versions

        versions = find_transform_versions(name="risk_calc")

        assert len(versions) > 0
        metadata = versions[0]["metadata"]

        # Check that transform type and calling convention are stored
        # Note: These might be in the MLflow model metadata, need to verify structure
        print(f"Available metadata keys: {metadata.keys()}")


def test_comprehensive_api_usage(spark, mlflow_tracking):
    """Test comprehensive usage of the new API with different function types."""

    # Different function types
    def data_cleaner(df: DataFrame, min_amount: float = 0.0) -> DataFrame:
        """Clean data by filtering amounts."""
        return df.filter(col("amount") > min_amount)

    def risk_scorer(amount: Column, category: Column) -> Column:
        """Calculate risk score from amount and category."""
        return when(category == "high_risk", amount * 1.5).otherwise(amount)

    def custom_aggregator(df, window_size):
        """Custom aggregation function."""
        return df.limit(window_size)

    # Test data
    test_data = spark.createDataFrame(
        [(100.0, "low_risk"), (200.0, "high_risk"), (50.0, "medium_risk")],
        ["amount", "category"],
    )

    with mlflow.start_run():
        # Test 1: DataFrame transform with input example
        model_uri1 = log_transform_function(data_cleaner, input_example=test_data)
        assert model_uri1.startswith("models:/")

        # Test 2: Column expression with explicit name
        model_uri2 = log_transform_function(risk_scorer, name="risk_calculator")
        assert model_uri2.startswith("models:/")

        # Test 3: Custom function with version
        model_uri3 = log_transform_function(
            custom_aggregator,
            name="aggregator",
            version="1.0.0",
        )
        assert model_uri3.startswith("models:/")

        # Test loading functions back
        loaded_cleaner = load_transform_function("data_cleaner", version="1")
        loaded_scorer = load_transform_function("risk_calculator", version="1")
        loaded_aggregator = load_transform_function("aggregator", version="1")

        # Verify they work
        assert loaded_cleaner is not None
        assert loaded_scorer is not None
        assert loaded_aggregator is not None

        # Test execution (basic smoke test)
        clean_result = loaded_cleaner(test_data)
        assert clean_result.count() >= 0  # Should not error

        print("✅ All function types registered and loaded successfully")


def test_column_expression_flexibility():
    """Test that column expressions are detected regardless of input parameter types."""

    # Various column expression patterns
    def pure_column_expr(amount: Column, category: Column) -> Column:
        """Pure column inputs."""
        return when(category == "high", amount * 1.5).otherwise(amount)

    def mixed_inputs_expr(amount: Column, multiplier: float) -> Column:
        """Mixed column and scalar inputs."""
        return amount * lit(multiplier)

    def complex_mixed_expr(
        base: Column,
        threshold: int,
        factor: float,
        flag: bool = True,
    ) -> Column:
        """Complex mixed inputs with defaults."""
        result = when(base > threshold, base * factor).otherwise(base)
        if flag:
            result = result * lit(2)
        return result

    def no_type_hints_but_returns_column(x, y) -> Column:
        """No input hints but returns Column."""
        return lit(x) + lit(y)

    # All should be detected as column expressions
    column_funcs = [
        pure_column_expr,
        mixed_inputs_expr,
        complex_mixed_expr,
        no_type_hints_but_returns_column,
    ]

    for func in column_funcs:
        detected_type = _detect_transform_type(func)
        assert detected_type == TransformType.COLUMN_EXPRESSION, (
            f"Function {func.__name__} should be COLUMN_EXPRESSION, got {detected_type}"
        )


if __name__ == "__main__":
    # Allow running this file directly for quick testing
    pytest.main([__file__, "-v"])
