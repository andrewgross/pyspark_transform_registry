#!/usr/bin/env python3
"""
Example demonstrating schema validation and MLflow storage in PySpark Transform Registry.

This script shows:
1. How schema constraints are automatically inferred
2. How they're stored in MLflow with the function
3. How runtime validation works when loading functions
4. How to inspect stored schema metadata
"""

import tempfile
import mlflow
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, lit, when

from pyspark_transform_registry import register_function, load_function
from pyspark_transform_registry.schema_constraints import (
    PartialSchemaConstraint,
    ColumnRequirement,
)


def create_spark_session():
    """Create a local Spark session for the example."""
    return (
        SparkSession.builder.appName("SchemaValidationExample")
        .master("local[2]")
        .config("spark.sql.warehouse.dir", tempfile.mkdtemp())
        .getOrCreate()
    )


def example_transform_function(df: DataFrame) -> DataFrame:
    """
    Example PySpark transform that processes customer data.

    This function demonstrates automatic schema inference:
    - Requires: customer_id (integer), amount (double), status (string)
    - Adds: risk_level (string), processed_date (string)
    - Preserves: other columns are kept
    """
    return (
        df.filter(col("amount") > 0)  # Requires 'amount' column (double)
        .filter(col("status").isNotNull())  # Requires 'status' column (string)
        .withColumn(
            "risk_level",  # Adds 'risk_level' column
            when(col("amount") > 1000, "high")
            .when(col("amount") > 100, "medium")
            .otherwise("low"),
        )
        .withColumn("processed_date", lit("2023-01-01"))  # Adds 'processed_date' column
        .select(
            "customer_id",
            "amount",
            "status",
            "risk_level",
            "processed_date",
        )  # Requires 'customer_id'
    )


def main():
    print("🚀 PySpark Transform Registry - Schema Validation Example")
    print("=" * 60)

    # Setup
    spark = create_spark_session()

    with tempfile.TemporaryDirectory() as temp_dir:
        # Configure MLflow to use temporary directory
        mlflow.set_tracking_uri(f"file://{temp_dir}")

        print("\n1️⃣ REGISTERING FUNCTION WITH AUTOMATIC SCHEMA INFERENCE")
        print("-" * 50)

        # Create sample data for signature inference
        sample_data = spark.createDataFrame(
            [(1, 150.0, "active"), (2, 2500.0, "pending"), (3, 75.0, "active")],
            ["customer_id", "amount", "status"],
        )

        print("📊 Sample input data:")
        sample_data.show()

        # Register function - schema will be automatically inferred
        model_uri = register_function(
            func=example_transform_function,
            name="customer.processing.risk_assessment",
            input_example=sample_data,
            description="Customer risk assessment with automatic schema validation",
            infer_schema=True,  # This enables automatic schema inference
        )

        print(f"✅ Function registered: {model_uri}")

        print("\n2️⃣ EXAMINING STORED SCHEMA METADATA IN MLFLOW")
        print("-" * 50)

        # Get the MLflow run details to show stored metadata
        client = mlflow.tracking.MlflowClient()
        registered_model = client.get_registered_model(
            "customer.processing.risk_assessment",
        )
        latest_version = registered_model.latest_versions[0]
        run = client.get_run(latest_version.run_id)

        print("🏷️  MLflow Run Tags (Schema Metadata):")
        schema_tags = {
            k: v for k, v in run.data.tags.items() if k.startswith("schema_")
        }
        for key, value in schema_tags.items():
            print(f"   {key}: {value}")

        # Parse and display the full schema constraint
        if "schema_constraint" in run.data.tags:
            constraint_json = run.data.tags["schema_constraint"]
            constraint = PartialSchemaConstraint.from_json(constraint_json)

            print("\n📋 Parsed Schema Constraint:")
            print(f"   Analysis Method: {constraint.analysis_method}")
            print(f"   Preserves Other Columns: {constraint.preserves_other_columns}")

            print(f"\n   Required Columns ({len(constraint.required_columns)}):")
            for col_req in constraint.required_columns:
                print(f"      - {col_req.name}: {col_req.type}")

            print(f"\n   Added Columns ({len(constraint.added_columns)}):")
            for col_add in constraint.added_columns:
                print(
                    f"      - {col_add.name}: {col_add.type} (operation: {col_add.operation})",
                )

            if constraint.warnings:
                print("\n   ⚠️  Warnings:")
                for warning in constraint.warnings:
                    print(f"      - {warning}")

        print("\n3️⃣ LOADING FUNCTION WITH VALIDATION ENABLED")
        print("-" * 50)

        # Load function with validation enabled (default behavior)
        loaded_function = load_function(
            "customer.processing.risk_assessment",
            version=1,
            validate_input=True,
            strict_validation=False,  # Warnings don't fail execution
        )

        print("✅ Function loaded with validation enabled")

        print("\n4️⃣ TESTING VALIDATION WITH VALID DATA")
        print("-" * 50)

        # Test with valid data (should pass validation)
        valid_data = spark.createDataFrame(
            [(10, 500.0, "active"), (11, 1500.0, "pending"), (12, 50.0, "active")],
            ["customer_id", "amount", "status"],
        )

        print("📊 Valid input data:")
        valid_data.show()

        print("🔍 Running validation and transformation...")
        result = loaded_function(valid_data)

        print("✅ Validation passed! Transformation result:")
        result.show()

        print("\n5️⃣ TESTING VALIDATION WITH INVALID DATA")
        print("-" * 50)

        # Test with invalid data (missing required column)
        invalid_data = spark.createDataFrame(
            [
                (20, 300.0),  # Missing 'status' column
                (21, 800.0),
            ],
            ["customer_id", "amount"],
        )

        print("📊 Invalid input data (missing 'status' column):")
        invalid_data.show()

        print("🔍 Testing validation with invalid data...")
        try:
            result = loaded_function(invalid_data)
            print("❌ Validation should have failed!")
        except ValueError as e:
            print(f"✅ Validation correctly failed: {e}")
        except Exception as e:
            print(
                f"✅ Function failed as expected (PySpark error): {type(e).__name__}: {e}",
            )

        print("\n6️⃣ LOADING WITH VALIDATION DISABLED")
        print("-" * 50)

        # Load function with validation disabled
        no_validation_function = load_function(
            "customer.processing.risk_assessment",
            version=1,
            validate_input=False,
        )

        print("📝 Loading function with validation disabled...")
        print("🔍 Testing with invalid data (validation disabled)...")

        try:
            result = no_validation_function(invalid_data)
            print("❌ Function should have failed due to missing column!")
        except Exception as e:
            print(f"✅ Function failed as expected (PySpark error): {type(e).__name__}")
            print("   This is normal - PySpark throws error when column doesn't exist")

        print("\n7️⃣ CUSTOM SCHEMA CONSTRAINTS")
        print("-" * 50)

        # Example of registering with custom schema constraints
        custom_constraint = PartialSchemaConstraint(
            required_columns=[
                ColumnRequirement("order_id", "integer"),
                ColumnRequirement("total_amount", "double"),
                ColumnRequirement("currency", "string"),
            ],
            added_columns=[
                ColumnRequirement("tax_amount", "double"),
                ColumnRequirement("final_total", "double"),
            ],
            preserves_other_columns=True,
            analysis_method="manual_specification",
        )

        def order_processor(df: DataFrame) -> DataFrame:
            """Process orders with tax calculation."""
            return df.withColumn("tax_amount", col("total_amount") * 0.1).withColumn(
                "final_total",
                col("total_amount") + col("tax_amount"),
            )

        register_function(
            func=order_processor,
            name="orders.processing.tax_calculator",
            schema_constraint=custom_constraint,  # Use custom constraint instead of inference
            description="Order processing with custom schema constraints",
        )

        print("✅ Function registered with custom schema constraints")

        # Show the custom constraint details
        custom_model = client.get_registered_model("orders.processing.tax_calculator")
        custom_version = custom_model.latest_versions[0]
        custom_run = client.get_run(custom_version.run_id)

        print("📋 Custom Schema Constraint Metadata:")
        print(
            f"   Analysis Method: {custom_run.data.tags.get('schema_analysis_method', 'N/A')}",
        )
        print(
            f"   Required Columns: {custom_run.data.tags.get('schema_required_columns', 'N/A')}",
        )
        print(
            f"   Added Columns: {custom_run.data.tags.get('schema_added_columns', 'N/A')}",
        )

        print("\n🎉 SCHEMA VALIDATION EXAMPLE COMPLETED!")
        print("=" * 60)

        spark.stop()


if __name__ == "__main__":
    main()
