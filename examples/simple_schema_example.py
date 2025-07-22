#!/usr/bin/env python3
"""
Simple example showing schema validation and MLflow storage.
"""

import json
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, lit, when

from pyspark_transform_registry.static_analysis import analyze_function
from pyspark_transform_registry.schema_constraints import PartialSchemaConstraint


def customer_risk_assessment(df: DataFrame) -> DataFrame:
    """
    Process customer data and assign risk levels.

    This function:
    - Requires: customer_id (integer), amount (double), status (string)
    - Adds: risk_level (string), processed_date (string)
    - Filters: amount > 0 and status is not null
    """
    return (
        df.filter(col("amount") > 0)
        .filter(col("status").isNotNull())
        .withColumn(
            "risk_level",
            when(col("amount") > 1000, "high")
            .when(col("amount") > 100, "medium")
            .otherwise("low"),
        )
        .withColumn("processed_date", lit("2023-01-01"))
        .select("customer_id", "amount", "status", "risk_level", "processed_date")
    )


def main():
    print("🔍 SCHEMA CONSTRAINT ANALYSIS EXAMPLE")
    print("=" * 50)

    # Analyze the function to extract schema constraints
    print("\n1️⃣ ANALYZING FUNCTION FOR SCHEMA CONSTRAINTS")
    constraint = analyze_function(customer_risk_assessment)

    print("✅ Analysis completed!")
    print(f"   Analysis Method: {constraint.analysis_method}")
    print(f"   Preserves Other Columns: {constraint.preserves_other_columns}")

    print(f"\n📋 REQUIRED COLUMNS ({len(constraint.required_columns)}):")
    for col_req in constraint.required_columns:
        print(f"   - {col_req.name}: {col_req.type}")

    print(f"\n📝 ADDED COLUMNS ({len(constraint.added_columns)}):")
    for col_add in constraint.added_columns:
        print(f"   - {col_add.name}: {col_add.type}")
        if hasattr(col_add, "operation") and col_add.operation:
            print(f"     Operation: {col_add.operation}")

    if constraint.warnings:
        print("\n⚠️  WARNINGS:")
        for warning in constraint.warnings:
            print(f"   - {warning}")

    print("\n2️⃣ JSON SERIALIZATION (How it's stored in MLflow)")
    print("-" * 50)

    # Show how the constraint is serialized for MLflow storage
    constraint_json = constraint.to_json()
    print("📦 Serialized constraint (stored as MLflow tag):")

    # Pretty print the JSON for readability
    constraint_dict = json.loads(constraint_json)
    print(json.dumps(constraint_dict, indent=2))

    print("\n3️⃣ MLFLOW METADATA TAGS")
    print("-" * 50)

    # Show what gets stored as MLflow tags
    print("🏷️  MLflow tags that would be stored:")
    print(f"   schema_constraint: {constraint_json[:100]}...")  # Truncated for display
    print(f"   schema_analysis_method: {constraint.analysis_method}")
    print(f"   schema_required_columns: {len(constraint.required_columns)}")
    print(f"   schema_added_columns: {len(constraint.added_columns)}")
    print(f"   schema_preserves_others: {constraint.preserves_other_columns}")

    if constraint.warnings:
        warnings_str = "; ".join(constraint.warnings)
        print(f"   schema_warnings: {warnings_str}")

    print("\n4️⃣ DESERIALIZATION (How it's loaded from MLflow)")
    print("-" * 50)

    # Show how the constraint is loaded back from MLflow
    loaded_constraint = PartialSchemaConstraint.from_json(constraint_json)

    print("✅ Constraint loaded from JSON successfully!")
    print(f"   Loaded required columns: {len(loaded_constraint.required_columns)}")
    print(f"   Loaded added columns: {len(loaded_constraint.added_columns)}")

    # Verify they match
    print("\n🔍 VERIFICATION:")
    print(
        f"   Required columns match: {len(constraint.required_columns) == len(loaded_constraint.required_columns)}",
    )
    print(
        f"   Added columns match: {len(constraint.added_columns) == len(loaded_constraint.added_columns)}",
    )

    print("\n5️⃣ VALIDATION LOGIC EXAMPLE")
    print("-" * 50)

    print("🔍 How validation would work:")
    print("\n✅ VALID INPUT:")
    print(
        "   DataFrame with columns: [customer_id: int, amount: double, status: string]",
    )
    print("   → Validation: PASS (has all required columns)")
    print("   → Function execution: SUCCESS")

    print("\n❌ INVALID INPUT:")
    print(
        "   DataFrame with columns: [customer_id: int, amount: double]",
    )  # Missing 'status'
    print("   → Validation: FAIL (missing 'status' column)")
    print("   → Error: Input validation failed: Missing required column 'status'")

    print("\n⚠️  PARTIAL MATCH:")
    print(
        "   DataFrame with columns: [customer_id: int, amount: string, status: string]",
    )  # Wrong type
    print("   → Validation: WARNING (amount should be double, got string)")
    print(
        "   → Function execution: May succeed or fail depending on PySpark type coercion",
    )

    print("\n🎉 SCHEMA VALIDATION EXAMPLE COMPLETE!")
    print("=" * 50)


if __name__ == "__main__":
    main()
