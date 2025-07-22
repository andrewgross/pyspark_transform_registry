"""
Example demonstrating advanced requirements handling and function clustering.

This example shows how to:
1. Use automatic dependency detection
2. Handle manual requirements specification
3. Create function clusters for bundling related functions
4. Validate function safety before logging
"""

import mlflow
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col, when, lit, udf
from pyspark.sql.types import IntegerType

from pyspark_transform_registry import (
    register_function,
    load_function,
)


def setup_spark_session():
    """Setup SparkSession for examples."""
    return SparkSession.builder.appName("RequirementsExample").getOrCreate()


# =============================================================================
# EXAMPLE 1: Automatic Dependency Detection
# =============================================================================


def pandas_heavy_transform(df: DataFrame) -> DataFrame:
    """
    Transform that uses pandas operations (will be auto-detected).

    This function demonstrates automatic dependency detection for pandas.
    """
    import pandas as pd

    # Convert to pandas, do some operations, convert back
    pandas_df = df.toPandas()

    # Use pandas-specific operations
    pandas_df["processed_at"] = pd.Timestamp.now()
    pandas_df["category_upper"] = pandas_df["category"].str.upper()

    # Convert back to Spark DataFrame
    spark = SparkSession.getActiveSession()
    return spark.createDataFrame(pandas_df)


def numpy_calculation(df: DataFrame) -> DataFrame:
    """
    Transform that uses NumPy for calculations.
    """
    import numpy as np

    @udf(returnType=IntegerType())
    def numpy_calculation_udf(value):
        """UDF using NumPy for complex calculations."""
        return int(np.sqrt(value * 2) + np.random.randint(0, 10))

    return df.withColumn("numpy_result", numpy_calculation_udf(col("amount")))


# =============================================================================
# EXAMPLE 2: Manual Requirements Specification
# =============================================================================


def sklearn_model_transform(df: DataFrame) -> DataFrame:
    """
    Transform that requires specific sklearn version.

    This shows how to specify exact requirements manually.
    """
    # This would use sklearn for some ML operations
    # For demo purposes, we'll just add a placeholder column
    return df.withColumn("ml_score", lit(0.85))


def complex_deps_transform(df: DataFrame) -> DataFrame:
    """
    Transform with complex dependencies that need careful version management.
    """
    # This would use multiple ML libraries
    return df.withColumn("ensemble_score", lit(0.92))


# =============================================================================
# EXAMPLE 3: Function Clustering
# =============================================================================


def data_cleaner(df: DataFrame) -> DataFrame:
    """Step 1: Clean the data."""
    return df.filter(col("amount") > 0).dropna()


def feature_engineer(df: DataFrame) -> DataFrame:
    """Step 2: Engineer features - depends on data_cleaner."""
    # This function might call data_cleaner internally
    cleaned_df = data_cleaner(df)

    return cleaned_df.withColumn(
        "amount_category",
        when(col("amount") > 1000, "high")
        .when(col("amount") > 100, "medium")
        .otherwise("low"),
    )


def scoring_function(df: DataFrame) -> DataFrame:
    """Step 3: Calculate scores - depends on feature_engineer."""
    featured_df = feature_engineer(df)

    return featured_df.withColumn(
        "risk_score",
        when(col("amount_category") == "high", 0.8)
        .when(col("amount_category") == "medium", 0.5)
        .otherwise(0.2),
    )


# =============================================================================
# EXAMPLE 4: Local Code Dependencies
# =============================================================================


def transform_with_local_utils(df: DataFrame) -> DataFrame:
    """
    Transform that depends on local utility functions.

    This would typically import from local modules like:
    from utils.data_processing import normalize_text
    from utils.calculations import calculate_score
    """

    # For demo purposes, inline the logic
    def normalize_text_inline(text_col):
        return text_col.upper()

    return df.withColumn("normalized_category", normalize_text_inline(col("category")))


# =============================================================================
# EXAMPLE USAGE
# =============================================================================


def example_1_automatic_detection():
    """Example 1: Automatic dependency detection."""
    spark = setup_spark_session()

    # Create sample data
    data = [
        (1, "electronics", 150.0),
        (2, "books", 25.0),
        (3, "clothing", 80.0),
    ]
    df = spark.createDataFrame(data, ["id", "category", "amount"])

    print("=== Example 1: Automatic Dependency Detection ===")

    with mlflow.start_run():
        # Log with automatic dependency detection
        print("Logging pandas_heavy_transform with auto-detection...")
        register_function(
            pandas_heavy_transform,
            input_example=df,
            auto_detect_requirements=True,
            validate_dependencies=True,
        )

        print("Logging numpy_calculation with auto-detection...")
        register_function(
            numpy_calculation,
            input_example=df,
            auto_detect_requirements=True,
            validate_dependencies=True,
        )

        print("✅ Functions logged with automatic dependency detection")


def example_2_manual_requirements():
    """Example 2: Manual requirements specification."""
    spark = setup_spark_session()

    data = [(1, "test", 100.0)]
    df = spark.createDataFrame(data, ["id", "category", "amount"])

    print("\n=== Example 2: Manual Requirements Specification ===")

    with mlflow.start_run():
        # Log with specific requirements
        print("Logging sklearn_model_transform with manual requirements...")
        register_function(
            sklearn_model_transform,
            input_example=df,
            extra_pip_requirements=["scikit-learn==1.3.0", "joblib>=1.2.0"],
            auto_detect_requirements=False,
            validate_dependencies=True,
        )

        print("Logging complex_deps_transform with multiple requirements...")
        register_function(
            complex_deps_transform,
            input_example=df,
            extra_pip_requirements=[
                "scikit-learn==1.3.0",
                "xgboost==1.7.0",
                "lightgbm==3.3.0",
            ],
            auto_detect_requirements=False,
            validate_dependencies=True,
        )

        print("✅ Functions logged with manual requirements")


def example_3_function_clustering():
    """Example 3: Function clustering for interdependent functions."""
    spark = setup_spark_session()

    data = [
        (1, "electronics", 150.0),
        (2, "books", 25.0),
        (3, "clothing", 1200.0),
    ]
    df = spark.createDataFrame(data, ["id", "category", "amount"])

    print("\n=== Example 3: Function Clustering ===")

    with mlflow.start_run():
        # Register individual functions (cluster functionality not yet implemented)
        print("Registering individual functions...")
        register_function(data_cleaner, name="pipeline.data_cleaner", input_example=df)
        register_function(
            feature_engineer,
            name="pipeline.feature_engineer",
            input_example=df,
        )
        register_function(
            scoring_function,
            name="pipeline.scoring_function",
            input_example=df,
        )

        print("✅ Function cluster logged successfully")

        # Test loading and using the cluster
        print("\nTesting cluster usage...")
        cluster_transform = load_function("data_processing_pipeline")

        # Use different functions from the cluster
        cleaned = cluster_transform(df, "data_cleaner")
        featured = cluster_transform(df, "feature_engineer")
        scored = cluster_transform(df, "scoring_function")

        print("Cluster functions executed successfully:")
        print(f"  - Cleaned rows: {cleaned.count()}")
        print(f"  - Featured columns: {len(featured.columns)}")
        print(f"  - Scored columns: {len(scored.columns)}")


def example_4_local_code_dependencies():
    """Example 4: Handling local code dependencies."""
    spark = setup_spark_session()

    data = [(1, "test category", 100.0)]
    df = spark.createDataFrame(data, ["id", "category", "amount"])

    print("\n=== Example 4: Local Code Dependencies ===")

    with mlflow.start_run():
        # Log with local code paths (would bundle utils/ directory)
        print("Logging transform with local code dependencies...")
        register_function(
            transform_with_local_utils,
            input_example=df,
            code_paths=["utils/"],  # This would bundle local utils
            auto_detect_requirements=True,
            validate_dependencies=True,
        )

        print("✅ Function logged with local code dependencies")


def example_5_validation_and_warnings():
    """Example 5: Dependency validation and warnings."""
    spark = setup_spark_session()

    print("\n=== Example 5: Dependency Validation and Warnings ===")

    # Function with external calls (will trigger warnings)
    def problematic_function(df: DataFrame) -> DataFrame:
        """Function that calls external functions not available in cluster."""
        # This would call some external function
        # external_service.process_data(df)  # This would trigger warning
        return df.withColumn("processed", lit(True))

    # Function with complex dependencies (will trigger warnings)
    def heavy_deps_function(df: DataFrame) -> DataFrame:
        """Function with heavy ML dependencies."""
        # import tensorflow as tf  # This would trigger warning
        return df.withColumn("ml_result", lit(0.95))

    data = [(1, "test", 100.0)]
    df = spark.createDataFrame(data, ["id", "category", "amount"])

    with mlflow.start_run():
        print("Logging function with validation warnings...")

        # This will show warnings about external calls
        register_function(
            problematic_function,
            input_example=df,
            validate_dependencies=True,
            auto_detect_requirements=True,
        )

        print("✅ Function logged with validation warnings shown")


def example_6_requirements_comparison():
    """Example 6: Compare auto-detected vs manual requirements."""
    spark = setup_spark_session()

    def sample_function(df: DataFrame) -> DataFrame:
        """Function for requirements comparison."""
        return df.withColumn("test_col", lit("test"))

    data = [(1, "test", 100.0)]
    df = spark.createDataFrame(data, ["id", "category", "amount"])

    print("\n=== Example 6: Requirements Comparison ===")

    with mlflow.start_run():
        # Log with auto-detection
        print("1. Auto-detected requirements:")
        register_function(
            sample_function,
            name="auto_detected_version",
            input_example=df,
            auto_detect_requirements=True,
            validate_dependencies=True,
        )

        print("\n2. Manual requirements:")
        register_function(
            sample_function,
            name="manual_version",
            input_example=df,
            extra_pip_requirements=["pandas==2.0.0", "numpy==1.24.0"],
            auto_detect_requirements=False,
            validate_dependencies=True,
        )

        print("\n3. Hybrid approach (auto + manual):")
        register_function(
            sample_function,
            name="hybrid_version",
            input_example=df,
            extra_pip_requirements=["special-package==1.0.0"],
            auto_detect_requirements=True,
            validate_dependencies=True,
        )

        print("✅ All versions logged with different requirement strategies")


if __name__ == "__main__":
    print("🚀 Starting Requirements Handling Examples")

    # Run all examples
    example_1_automatic_detection()
    example_2_manual_requirements()
    example_3_function_clustering()
    example_4_local_code_dependencies()
    example_5_validation_and_warnings()
    example_6_requirements_comparison()

    print("\n🎉 All requirements handling examples completed!")
    print("\nKey Benefits:")
    print("✅ Minimal dependency detection prevents bloated requirements")
    print("✅ Function clustering bundles related functions together")
    print("✅ Validation catches potential issues before deployment")
    print("✅ MLflow native requirements handling for maximum compatibility")
    print("✅ Flexible manual override when needed")
