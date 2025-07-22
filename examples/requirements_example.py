"""
Example demonstrating requirements handling in PySpark Transform Registry.

This example shows how to:
1. Use extra_pip_requirements for manual dependency specification
2. Register functions with specific package versions
3. Load and use functions with custom requirements
"""

import mlflow
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col, when, lit

from pyspark_transform_registry import (
    register_function,
    load_function,
)


def setup_spark_session():
    """Setup SparkSession for examples."""
    return SparkSession.builder.appName("RequirementsExample").getOrCreate()


# =============================================================================
# EXAMPLE 1: Basic Transform with Standard Dependencies
# =============================================================================


def basic_transform(df: DataFrame) -> DataFrame:
    """
    Basic transform using only standard PySpark functionality.
    """
    return df.withColumn(
        "risk_category",
        when(col("amount") > 1000, "high")
        .when(col("amount") > 100, "medium")
        .otherwise("low"),
    )


# =============================================================================
# EXAMPLE 2: Transform with Custom Requirements
# =============================================================================


def advanced_transform(df: DataFrame) -> DataFrame:
    """
    Transform that requires specific package versions.

    Note: This example shows how to specify requirements,
    but the actual packages may not be used in this simple example.
    """
    # In a real scenario, this might use pandas or numpy functionality
    return df.withColumn("processed_flag", lit(True))


def ml_scoring_transform(df: DataFrame) -> DataFrame:
    """
    Transform that would use ML libraries.

    Note: This example shows requirement specification for ML packages.
    """
    # In a real scenario, this might use scikit-learn or other ML libraries
    return df.withColumn("ml_score", lit(0.95))


# =============================================================================
# EXAMPLE USAGE
# =============================================================================


def example_1_basic_transform():
    """Example 1: Basic transform without custom requirements."""
    spark = setup_spark_session()

    # Create sample data
    data = [
        (1, "electronics", 150.0),
        (2, "books", 25.0),
        (3, "clothing", 80.0),
    ]
    df = spark.createDataFrame(data, ["id", "category", "amount"])

    print("=== Example 1: Basic Transform ===")
    print("Input data:")
    df.show()

    with mlflow.start_run():
        # Register with no custom requirements
        print("Registering basic_transform...")
        register_function(
            func=basic_transform,
            name="basic_transform",
            input_example=df,
            description="Basic risk categorization",
        )

        print("✅ Function registered successfully")

        # Load and test
        transform = load_function("basic_transform", version=1)
        result = transform(df)

        print("Result:")
        result.show()


def example_2_custom_requirements():
    """Example 2: Transform with custom requirements."""
    spark = setup_spark_session()

    data = [(1, "test", 100.0)]
    df = spark.createDataFrame(data, ["id", "category", "amount"])

    print("\n=== Example 2: Custom Requirements ===")

    with mlflow.start_run():
        # Register with specific requirements
        print("Registering advanced_transform with custom requirements...")
        register_function(
            func=advanced_transform,
            name="advanced_transform",
            input_example=df,
            extra_pip_requirements=["pandas>=2.0.0", "numpy>=1.24.0"],
            description="Advanced transform with custom dependencies",
        )

        print("✅ Function registered with custom requirements")

        # Load and test
        transform = load_function("advanced_transform", version=1)
        result = transform(df)

        print("Result:")
        result.show()


def example_3_ml_requirements():
    """Example 3: Transform with ML library requirements."""
    spark = setup_spark_session()

    data = [
        (1, "customer_a", 500.0),
        (2, "customer_b", 1200.0),
        (3, "customer_c", 75.0),
    ]
    df = spark.createDataFrame(data, ["id", "customer", "amount"])

    print("\n=== Example 3: ML Library Requirements ===")

    with mlflow.start_run():
        # Register with ML requirements
        print("Registering ml_scoring_transform with ML requirements...")
        register_function(
            func=ml_scoring_transform,
            name="ml_scoring_transform",
            input_example=df,
            extra_pip_requirements=[
                "scikit-learn>=1.3.0",
                "xgboost>=1.7.0",
                "lightgbm>=3.3.0",
            ],
            description="ML scoring transform with multiple ML dependencies",
        )

        print("✅ Function registered with ML requirements")

        # Load and test
        transform = load_function("ml_scoring_transform", version=1)
        result = transform(df)

        print("Result:")
        result.show()


def example_4_version_specific_requirements():
    """Example 4: Transform with version-specific requirements."""
    spark = setup_spark_session()

    def version_specific_transform(df: DataFrame) -> DataFrame:
        """Transform requiring specific package versions."""
        return df.withColumn("version_info", lit("v1.0"))

    data = [(1, "test", 50.0)]
    df = spark.createDataFrame(data, ["id", "name", "value"])

    print("\n=== Example 4: Version-Specific Requirements ===")

    with mlflow.start_run():
        # Register with very specific version requirements
        print("Registering transform with version-specific requirements...")
        register_function(
            func=version_specific_transform,
            name="version_specific_transform",
            input_example=df,
            extra_pip_requirements=[
                "requests==2.31.0",  # Exact version
                "urllib3>=1.26.0,<3.0.0",  # Version range
                "certifi~=2023.0",  # Compatible version
            ],
            description="Transform with specific version constraints",
        )

        print("✅ Function registered with version-specific requirements")

        # Load and test
        transform = load_function("version_specific_transform", version=1)
        result = transform(df)

        print("Result:")
        result.show()


def example_5_discovery_with_requirements():
    """Example 5: Discovering registered functions and their requirements."""
    print("\n=== Example 5: Model Discovery ===")

    # Use MLflow to discover registered models
    client = mlflow.tracking.MlflowClient()
    models = client.list_registered_models()

    print("📋 Registered models:")
    for model in models:
        print(f"\n🔍 Model: {model.name}")

        # Get latest version details
        latest_versions = client.get_latest_versions(model.name)
        for version in latest_versions:
            print(f"  📦 Version: {version.version}")

            # Get run details to see requirements
            try:
                run = client.get_run(version.run_id)
                if "description" in run.data.tags:
                    print(f"  📝 Description: {run.data.tags['description']}")

                # Note: Requirements are handled internally by MLflow
                # They're not directly visible as tags, but are used during model loading
                print("  ✅ Requirements handled by MLflow internally")

            except Exception as e:
                print(f"  ⚠️  Could not get run details: {e}")


if __name__ == "__main__":
    print("🚀 Starting Requirements Handling Examples")

    # Run all examples
    example_1_basic_transform()
    example_2_custom_requirements()
    example_3_ml_requirements()
    example_4_version_specific_requirements()
    example_5_discovery_with_requirements()

    print("\n🎉 All requirements handling examples completed!")
    print("\nKey Benefits:")
    print("✅ Specify exact package versions for reproducibility")
    print("✅ MLflow handles dependency management automatically")
    print("✅ Clear separation of transform logic from dependencies")
    print("✅ Version constraints ensure compatibility")
