"""
Examples of using the PySpark Transform Registry with varying complexity levels.

This module demonstrates:
1. Simple transforms with basic operations
2. Intermediate transforms with multiple operations
3. Complex transforms with custom logic and dependencies
4. Workflow examples showing how to chain and reuse transforms
"""

import mlflow
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import (
    col,
    when,
    lit,
    regexp_replace,
    split,
    length,
    upper,
    row_number,
)

from pyspark_transform_registry import (
    register_function,
    load_function,
    list_registered_functions,
)


def setup_spark_session():
    """Setup SparkSession for examples."""
    return SparkSession.builder.appName("TransformRegistryExamples").getOrCreate()


# =============================================================================
# 1. SIMPLE TRANSFORMS - Basic Column Operations
# =============================================================================


def add_sales_tax(df: DataFrame, tax_rate: float = 0.08) -> DataFrame:
    """
    Simple transform: Add sales tax column to a DataFrame.

    Args:
        df: Input DataFrame with 'price' column
        tax_rate: Tax rate to apply (default 8%)

    Returns:
        DataFrame with additional 'sales_tax' and 'total_price' columns
    """
    return df.withColumn("sales_tax", col("price") * tax_rate).withColumn(
        "total_price",
        col("price") + col("sales_tax"),
    )


def normalize_text(df: DataFrame, column_name: str) -> DataFrame:
    """
    Simple transform: Normalize text in a specified column.

    Args:
        df: Input DataFrame
        column_name: Name of column to normalize

    Returns:
        DataFrame with normalized text column
    """
    return df.withColumn(
        column_name,
        upper(regexp_replace(col(column_name), r"[^a-zA-Z0-9\s]", "")),
    )


# =============================================================================
# 2. INTERMEDIATE TRANSFORMS - Multiple Operations
# =============================================================================


def customer_segmentation(df: DataFrame) -> DataFrame:
    """
    Intermediate transform: Customer segmentation based on purchase behavior.

    Args:
        df: DataFrame with columns: customer_id, total_spent, purchase_frequency

    Returns:
        DataFrame with additional 'segment' column
    """
    return df.withColumn(
        "segment",
        when((col("total_spent") >= 1000) & (col("purchase_frequency") >= 5), "Premium")
        .when((col("total_spent") >= 500) & (col("purchase_frequency") >= 3), "Gold")
        .when((col("total_spent") >= 100) & (col("purchase_frequency") >= 1), "Silver")
        .otherwise("Bronze"),
    ).withColumn(
        "segment_score",
        when(col("segment") == "Premium", 4)
        .when(col("segment") == "Gold", 3)
        .when(col("segment") == "Silver", 2)
        .otherwise(1),
    )


def clean_and_validate_data(df: DataFrame) -> DataFrame:
    """
    Intermediate transform: Clean and validate customer data.

    Args:
        df: DataFrame with customer data

    Returns:
        Cleaned DataFrame with validation flags
    """
    return (
        df.withColumn(
            "email_valid",
            col("email").rlike(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"),
        )
        .withColumn("phone_cleaned", regexp_replace(col("phone"), r"[^\d]", ""))
        .withColumn("name_parts", split(col("name"), " "))
        .withColumn("first_name", col("name_parts")[0])
        .withColumn(
            "last_name",
            when(length(col("name_parts")) > 1, col("name_parts")[1]).otherwise(
                lit(""),
            ),
        )
        .drop("name_parts")
    )


# =============================================================================
# 3. COMPLEX TRANSFORMS - Custom Logic and Dependencies
# =============================================================================


def advanced_feature_engineering(df: DataFrame) -> DataFrame:
    """
    Complex transform: Advanced feature engineering for ML pipeline.

    Args:
        df: DataFrame with transaction data

    Returns:
        DataFrame with engineered features
    """
    from pyspark.sql.functions import (
        avg,
        count,
        max as spark_max,
        min as spark_min,
        stddev,
    )
    from pyspark.sql.window import Window

    # Window functions for customer-level aggregations
    customer_window = Window.partitionBy("customer_id")

    # Time-based window (last 30 days) - could be used for rolling calculations
    # time_window = (
    #     Window.partitionBy("customer_id")
    #     .orderBy("transaction_date")
    #     .rowsBetween(Window.unboundedPreceding, Window.currentRow)
    # )

    return (
        df.withColumn(
            "avg_transaction_amount",
            avg(col("amount")).over(customer_window),
        )
        .withColumn("transaction_count", count("*").over(customer_window))
        .withColumn(
            "max_transaction_amount",
            spark_max(col("amount")).over(customer_window),
        )
        .withColumn(
            "min_transaction_amount",
            spark_min(col("amount")).over(customer_window),
        )
        .withColumn("transaction_stddev", stddev(col("amount")).over(customer_window))
        .withColumn(
            "is_high_value_customer",
            when(col("avg_transaction_amount") > 500, True).otherwise(False),
        )
        .withColumn(
            "transaction_rank",
            row_number().over(
                Window.partitionBy("customer_id").orderBy(col("amount").desc()),
            ),
        )
    )


def ml_preprocessing_pipeline(df: DataFrame) -> DataFrame:
    """
    Complex transform: ML preprocessing pipeline with multiple stages.

    Args:
        df: Raw DataFrame for ML model

    Returns:
        Preprocessed DataFrame ready for ML training
    """
    from pyspark.sql.functions import when, col

    # Handle missing values
    df_clean = df.fillna({"age": 0, "income": 0, "credit_score": 600})

    # Feature scaling and normalization
    df_scaled = df_clean.withColumn(
        "income_scaled",
        (col("income") - 50000) / 25000,  # Simple z-score normalization
    ).withColumn(
        "age_binned",
        when(col("age") < 25, "young")
        .when(col("age") < 55, "middle")
        .otherwise("senior"),
    )

    # Feature interactions
    df_features = df_scaled.withColumn(
        "income_age_interaction",
        col("income_scaled") * col("age") / 100,
    ).withColumn(
        "credit_income_ratio",
        col("credit_score") / (col("income") + 1),  # Add 1 to avoid division by zero
    )

    # Add data quality flags
    df_final = df_features.withColumn(
        "data_quality_score",
        when(col("income") > 0, 1).otherwise(0)
        + when(col("age") > 0, 1).otherwise(0)
        + when(col("credit_score") > 300, 1).otherwise(0),
    )

    return df_final


# =============================================================================
# 4. WORKFLOW EXAMPLES - Chaining and Reusing Transforms
# =============================================================================


def example_1_simple_workflow():
    """Example 1: Simple workflow with basic transforms."""
    spark = setup_spark_session()

    # Create sample data
    data = [(1, "apple", 2.50), (2, "banana", 1.20), (3, "cherry", 3.00)]
    df = spark.createDataFrame(data, ["id", "product", "price"])

    print("=== Example 1: Simple Workflow ===")
    print("Original data:")
    df.show()

    # Start MLflow run
    with mlflow.start_run():
        # Log the transform
        register_function(add_sales_tax, "add_sales_tax")
        print("✓ Logged 'add_sales_tax' transform")

        # Apply the transform
        result = add_sales_tax(df, tax_rate=0.10)
        print("Result after applying sales tax:")
        result.show()

        # Load and reuse the transform
        loaded_transform = load_function("add_sales_tax")
        reloaded_result = loaded_transform(df, tax_rate=0.08)
        print("Result after reloading transform with different tax rate:")
        reloaded_result.show()


def example_2_intermediate_workflow():
    """Example 2: Intermediate workflow with customer segmentation."""
    spark = setup_spark_session()

    # Create sample customer data
    customer_data = [
        (1, 1500, 8),  # Premium
        (2, 750, 4),  # Gold
        (3, 200, 2),  # Silver
        (4, 50, 1),  # Bronze
    ]
    df = spark.createDataFrame(
        customer_data,
        ["customer_id", "total_spent", "purchase_frequency"],
    )

    print("\n=== Example 2: Intermediate Workflow ===")
    print("Customer data:")
    df.show()

    with mlflow.start_run():
        # Log the segmentation transform
        register_function(customer_segmentation, "customer_segmentation")
        print("✓ Logged 'customer_segmentation' transform")

        # Apply segmentation
        segmented_df = customer_segmentation(df)
        print("Customer segmentation results:")
        segmented_df.show()

        # Find all available transforms
        transforms = list_registered_functions()
        print(f"✓ Found {len(transforms)} available transforms")
        for transform in transforms:
            print(f"  - {transform['metadata']['transform_name']}")


def example_3_complex_workflow():
    """Example 3: Complex workflow with chained transforms."""
    spark = setup_spark_session()

    # Create sample transaction data
    transaction_data = [
        (1, 1, "2023-01-01", 100.0),
        (2, 1, "2023-01-15", 250.0),
        (3, 2, "2023-01-10", 75.0),
        (4, 2, "2023-01-20", 150.0),
        (5, 3, "2023-01-05", 500.0),
    ]
    df = spark.createDataFrame(
        transaction_data,
        ["transaction_id", "customer_id", "transaction_date", "amount"],
    )

    print("\n=== Example 3: Complex Workflow ===")
    print("Transaction data:")
    df.show()

    with mlflow.start_run():
        # Log the complex transform
        register_function(advanced_feature_engineering, "advanced_features")
        print("✓ Logged 'advanced_features' transform")

        # Apply feature engineering
        featured_df = advanced_feature_engineering(df)
        print("Feature engineering results:")
        featured_df.show()


def example_4_full_ml_pipeline():
    """Example 4: Full ML pipeline with multiple chained transforms."""
    spark = setup_spark_session()

    # Create sample ML data
    ml_data = [
        (1, 25, 45000, 650, "john@email.com", "123-456-7890", "John Doe"),
        (2, 35, 75000, 720, "jane@email.com", "987-654-3210", "Jane Smith"),
        (3, 45, 95000, 800, "bob@email.com", "555-123-4567", "Bob Johnson"),
        (4, 55, 120000, 750, "alice@email.com", "444-987-6543", "Alice Brown"),
    ]
    df = spark.createDataFrame(
        ml_data,
        ["id", "age", "income", "credit_score", "email", "phone", "name"],
    )

    print("\n=== Example 4: Full ML Pipeline ===")
    print("Raw ML data:")
    df.show()

    with mlflow.start_run():
        # Log all transforms
        register_function(clean_and_validate_data, "data_cleaning")
        register_function(ml_preprocessing_pipeline, "ml_preprocessing")
        print("✓ Logged data cleaning and ML preprocessing transforms")

        # Step 1: Clean and validate data
        cleaned_df = clean_and_validate_data(df)
        print("After data cleaning:")
        cleaned_df.show()

        # Step 2: Apply ML preprocessing
        processed_df = ml_preprocessing_pipeline(cleaned_df)
        print("After ML preprocessing:")
        processed_df.show()

        # Step 3: Show how to reload and chain transforms
        print("\n--- Reloading and chaining transforms ---")
        cleaning_transform = load_function("data_cleaning")
        ml_transform = load_function("ml_preprocessing")

        # Chain the transforms
        pipeline_result = ml_transform(cleaning_transform(df))
        print("Pipeline result using loaded transforms:")
        pipeline_result.show()


def example_5_version_management():
    """Example 5: Version management and transform discovery."""
    spark = setup_spark_session()

    print("\n=== Example 5: Version Management ===")

    # Create multiple versions of a transform
    def calculate_discount_v1(df: DataFrame) -> DataFrame:
        """Version 1: Simple 10% discount."""
        return df.withColumn("discount", col("price") * 0.1)

    def calculate_discount_v2(df: DataFrame) -> DataFrame:
        """Version 2: Tiered discount based on price."""
        return df.withColumn(
            "discount",
            when(col("price") > 100, col("price") * 0.15)
            .when(col("price") > 50, col("price") * 0.10)
            .otherwise(col("price") * 0.05),
        )

    # Sample data
    data = [(1, "item1", 25.0), (2, "item2", 75.0), (3, "item3", 150.0)]
    df = spark.createDataFrame(data, ["id", "name", "price"])

    with mlflow.start_run():
        # Log version 1
        register_function(
            calculate_discount_v1,
            "calculate_discount",
            version="1.0.0",
        )
        print("✓ Logged calculate_discount version 1.0.0")

        # Log version 2
        register_function(
            calculate_discount_v2,
            "calculate_discount",
            version="2.0.0",
        )
        print("✓ Logged calculate_discount version 2.0.0")

        # Find all versions
        versions = list_registered_functions(name="calculate_discount")
        print(f"Found {len(versions)} versions of calculate_discount:")
        for v in versions:
            print(f"  - Version {v['semantic_version']}")

        # Load and compare different versions
        v1_transform = load_function(
            "calculate_discount",
            version_constraint=">=1.0.0,<2.0.0",
        )
        v2_transform = load_function(
            "calculate_discount",
            version_constraint=">=2.0.0",
        )

        print("\nComparison of different versions:")
        print("Version 1.0.0 results:")
        v1_transform(df).show()

        print("Version 2.0.0 results:")
        v2_transform(df).show()


if __name__ == "__main__":
    # Run all examples
    example_1_simple_workflow()
    example_2_intermediate_workflow()
    example_3_complex_workflow()
    example_4_full_ml_pipeline()
    example_5_version_management()

    print("\n🎉 All examples completed successfully!")
