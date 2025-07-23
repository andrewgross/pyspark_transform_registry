#!/usr/bin/env python3
"""
Example demonstrating source code inspection functionality.

This example shows how to:
1. Register a function and load it
2. Access the original function source code
3. Get the unwrapped original function for advanced inspection
4. Compare wrapper vs original function inspection
"""

import tempfile

import mlflow
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col, lit, when

from pyspark_transform_registry import load_function, register_function


def create_spark_session():
    """Create a local Spark session for the example."""
    return (
        SparkSession.builder.appName("SourceInspectionExample")
        .master("local[2]")
        .config("spark.sql.warehouse.dir", tempfile.mkdtemp())
        .getOrCreate()
    )


def risk_assessment_transform(df: DataFrame) -> DataFrame:
    """
    Assess risk levels for financial transactions.

    This function categorizes transactions based on amount:
    - High risk: > $10,000
    - Medium risk: $1,000 - $10,000
    - Low risk: < $1,000

    Args:
        df: DataFrame with 'amount' column

    Returns:
        DataFrame with additional 'risk_level' and 'requires_approval' columns
    """
    return (
        df.withColumn(
            "risk_level",
            when(col("amount") > 10000, "high")
            .when(col("amount") > 1000, "medium")
            .otherwise("low"),
        )
        .withColumn("requires_approval", col("risk_level").isin("high", "medium"))
        .withColumn("processed_timestamp", lit("2024-01-01 12:00:00"))
    )


def main():
    print("🔍 Source Code Inspection Example")
    print("=" * 50)

    spark = create_spark_session()

    with tempfile.TemporaryDirectory() as temp_dir:
        # Configure MLflow
        mlflow.set_tracking_uri(f"file://{temp_dir}")
        experiment_id = mlflow.create_experiment("source_inspection_demo")
        mlflow.set_experiment(experiment_id=experiment_id)

        # Sample transaction data
        transactions = spark.createDataFrame(
            [
                (1, "TXN001", 500.0),
                (2, "TXN002", 5000.0),
                (3, "TXN003", 15000.0),
                (4, "TXN004", 250.0),
            ],
            ["id", "transaction_id", "amount"],
        )

        print("\n1️⃣ REGISTERING FUNCTION")
        print("-" * 30)

        with mlflow.start_run():
            model_uri = register_function(
                func=risk_assessment_transform,
                name="risk_assessment",
                input_example=transactions,
                description="Financial transaction risk assessment",
            )
            print(f"✅ Function registered: {model_uri}")

        print("\n2️⃣ LOADING AND TESTING FUNCTION")
        print("-" * 30)

        # Load the function
        risk_assessor = load_function("risk_assessment", version=1)
        print("✅ Function loaded successfully")

        # Test the function
        result = risk_assessor(transactions)
        print("\n📊 Function execution result:")
        result.show()

        print("\n3️⃣ SOURCE CODE INSPECTION")
        print("-" * 30)

        # Get the original source code
        print("📄 Original function source code:")
        source_code = risk_assessor.get_source()
        print("\n" + "─" * 60)
        print(source_code)
        print("─" * 60)

        print("\n4️⃣ FUNCTION METADATA INSPECTION")
        print("-" * 30)

        # Get the original function for detailed inspection
        original_func = risk_assessor.get_original_function()

        print(f"📝 Function name: {original_func.__name__}")
        print(f"📖 Function module: {original_func.__module__}")

        # Get function docstring
        docstring = original_func.__doc__
        if docstring:
            print("📚 Function docstring:")
            print(docstring.strip())

        # Get function signature
        import inspect

        signature = inspect.signature(original_func)
        print(f"\n🔍 Function signature: {original_func.__name__}{signature}")

        # Get parameter details
        print("\n📋 Parameters:")
        for name, param in signature.parameters.items():
            print(
                f"  - {name}: {param.annotation.__name__ if param.annotation != param.empty else 'Any'}",
            )

        # Get return type
        return_annotation = signature.return_annotation
        if return_annotation != signature.empty:
            print(f"\n↩️  Return type: {return_annotation.__name__}")

        print("\n5️⃣ COMPARISON: WRAPPER VS ORIGINAL")
        print("-" * 30)

        # Show the difference between inspecting wrapper vs original
        print("🔄 When you use inspect.getsource() on the loaded function:")
        print("   → You get the wrapper function code (internal implementation)")

        print("\n🎯 When you use .get_source() on the loaded function:")
        print("   → You get the original registered function code")

        try:
            wrapper_source = inspect.getsource(risk_assessor)
            wrapper_lines = len(wrapper_source.split("\n"))
            original_lines = len(source_code.split("\n"))

            print("\n📊 Comparison:")
            print(f"   Wrapper code: {wrapper_lines} lines")
            print(f"   Original code: {original_lines} lines")
            print(f"   Ratio: {wrapper_lines / original_lines:.1f}x larger")

        except Exception as e:
            print(f"\n⚠️  Could not inspect wrapper: {e}")

        print("\n6️⃣ PRACTICAL USE CASES")
        print("-" * 30)

        print("🛠️  Source inspection is useful for:")
        print("   • Debugging registered functions")
        print("   • Code review and auditing")
        print("   • Understanding function logic")
        print("   • Generating documentation")
        print("   • Validating function behavior")

        print("\n💡 Example: Generate function documentation")
        func_info = {
            "name": original_func.__name__,
            "docstring": original_func.__doc__,
            "signature": str(signature),
            "source_lines": len(source_code.split("\n")),
        }

        print("📄 Auto-generated function info:")
        for key, value in func_info.items():
            if key == "docstring" and value:
                print(
                    f"   {key}: {repr(value[:50] + '...' if len(str(value)) > 50 else value)}",
                )
            else:
                print(f"   {key}: {value}")

    print("\n🎉 SOURCE INSPECTION EXAMPLE COMPLETED!")
    spark.stop()


if __name__ == "__main__":
    main()
    main()
    main()
