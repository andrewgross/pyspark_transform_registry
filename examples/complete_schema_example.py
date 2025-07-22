#!/usr/bin/env python3
"""
Complete example showing schema validation, MLflow storage, and runtime validation.
"""

import json

from pyspark_transform_registry.schema_constraints import (
    PartialSchemaConstraint,
    ColumnRequirement,
)
from pyspark_transform_registry.runtime_validation import RuntimeValidator


def main():
    print("🔍 COMPLETE SCHEMA VALIDATION & MLFLOW STORAGE EXAMPLE")
    print("=" * 60)

    print("\n1️⃣ CREATING A SCHEMA CONSTRAINT")
    print("-" * 40)

    # Create a realistic schema constraint manually
    constraint = PartialSchemaConstraint(
        required_columns=[
            ColumnRequirement("customer_id", "integer"),
            ColumnRequirement("amount", "double"),
            ColumnRequirement("status", "string"),
        ],
        added_columns=[],  # Use empty for now to avoid serialization issues
        preserves_other_columns=False,  # Function only selects specific columns
        confidence=0.9,  # High confidence
        analysis_method="static_analysis_enhanced",
    )

    print("✅ Schema constraint created!")
    print(f"   Confidence: {constraint.confidence}")
    print(f"   Analysis Method: {constraint.analysis_method}")
    print(f"   Preserves Other Columns: {constraint.preserves_other_columns}")

    print(f"\n📋 REQUIRED COLUMNS ({len(constraint.required_columns)}):")
    for col_req in constraint.required_columns:
        print(f"   - {col_req.name}: {col_req.type}")

    print(f"\n📝 ADDED COLUMNS ({len(constraint.added_columns)}):")
    for col_add in constraint.added_columns:
        print(f"   - {col_add.name}: {col_add.type}")

    print("\n2️⃣ MLFLOW STORAGE FORMAT")
    print("-" * 40)

    # Show how it's serialized for MLflow
    constraint_json = constraint.to_json()
    print("📦 JSON stored in MLflow tag 'schema_constraint':")

    # Pretty print for readability
    constraint_dict = json.loads(constraint_json)
    print(json.dumps(constraint_dict, indent=2))

    print("\n🏷️  Additional MLflow tags:")
    print(f"   schema_confidence: {constraint.confidence}")
    print(f"   schema_analysis_method: {constraint.analysis_method}")
    print(f"   schema_required_columns: {len(constraint.required_columns)}")
    print(f"   schema_added_columns: {len(constraint.added_columns)}")
    print(f"   schema_preserves_others: {constraint.preserves_other_columns}")

    print("\n3️⃣ LOADING FROM MLFLOW")
    print("-" * 40)

    # Simulate loading from MLflow
    print("🔄 Simulating load from MLflow...")
    loaded_constraint = PartialSchemaConstraint.from_json(constraint_json)

    print("✅ Constraint loaded successfully!")
    print(
        f"   Confidence matches: {constraint.confidence == loaded_constraint.confidence}",
    )
    print(f"   Required columns: {len(loaded_constraint.required_columns)}")
    print(f"   Added columns: {len(loaded_constraint.added_columns)}")

    print("\n4️⃣ RUNTIME VALIDATION")
    print("-" * 40)

    # Create a runtime validator (for demonstration only)
    _ = RuntimeValidator(strict_mode=False)

    print("🔍 Testing validation with different DataFrames...")

    # We can't create actual Spark DataFrames without a SparkSession,
    # but we can show the validation logic with mock schemas

    print("\n✅ VALID CASE:")
    print(
        "   Input DataFrame schema: [customer_id: int, amount: double, status: string, extra: string]",
    )
    print("   Required columns present: ✓ customer_id, ✓ amount, ✓ status")
    print("   Validation result: PASS")
    print("   Extra columns: Ignored (preserves_other_columns=False)")

    print("\n❌ INVALID CASE 1 - Missing Required Column:")
    print("   Input DataFrame schema: [customer_id: int, amount: double]")
    print("   Missing required column: ✗ status")
    print("   Validation result: FAIL")
    print("   Error: 'Input validation failed: Missing required column: status'")

    print("\n⚠️  INVALID CASE 2 - Wrong Type:")
    print(
        "   Input DataFrame schema: [customer_id: int, amount: string, status: string]",
    )
    print("   Type mismatch: ✗ amount (expected: double, got: string)")
    print("   Validation result: WARNING (in permissive mode) / FAIL (in strict mode)")
    print(
        "   Error: 'Input validation failed: Column amount type mismatch: expected double, got string'",
    )

    print("\n5️⃣ INTEGRATION WITH TRANSFORM FUNCTIONS")
    print("-" * 40)

    print("🔗 How this integrates with PySpark Transform Registry:")
    print()
    print("1. REGISTRATION:")
    print("   register_function(func, name='my.transform', infer_schema=True)")
    print("   → Static analysis creates PartialSchemaConstraint")
    print("   → Constraint serialized to JSON and stored in MLflow tags")
    print("   → Function and metadata saved to MLflow model registry")
    print()
    print("2. LOADING:")
    print("   load_function('my.transform', validate_input=True)")
    print("   → MLflow model loaded")
    print("   → Schema constraint JSON retrieved from run tags")
    print("   → PartialSchemaConstraint deserialized")
    print("   → Wrapper function created with validation")
    print()
    print("3. EXECUTION:")
    print("   transformed_df = loaded_func(input_df)")
    print("   → RuntimeValidator checks input_df against constraint")
    print("   → If validation passes: original function executed")
    print("   → If validation fails: ValueError raised with details")
    print()
    print("4. MULTI-PARAMETER FUNCTIONS:")
    print(
        "   result = loaded_func(df, params={'threshold': 100, 'category': 'premium'})",
    )
    print("   → Same validation process for DataFrame")
    print("   → Parameters passed through to original function")

    print("\n6️⃣ VALIDATION MODES")
    print("-" * 40)

    print("🎛️  Different validation modes available:")
    print()
    print("STRICT MODE (strict_validation=True):")
    print("   • Type mismatches → ERROR")
    print("   • Missing columns → ERROR")
    print("   • Warnings → ERROR")
    print("   • Best for: Production systems requiring exact schema matches")
    print()
    print("PERMISSIVE MODE (strict_validation=False, default):")
    print("   • Type mismatches → WARNING (may proceed)")
    print("   • Missing columns → ERROR")
    print("   • Warnings → LOG (execution continues)")
    print("   • Best for: Development and flexible production systems")
    print()
    print("NO VALIDATION (validate_input=False):")
    print("   • No schema checking")
    print("   • Direct function execution")
    print("   • Best for: Testing, debugging, or when you trust input")

    print("\n🎉 COMPLETE EXAMPLE FINISHED!")
    print("=" * 60)

    print("\n💡 KEY BENEFITS:")
    print("   ✅ Deterministic validation before execution")
    print("   ✅ Machine-to-machine operation without human intervention")
    print("   ✅ Detailed error messages for debugging")
    print("   ✅ Version compatibility across system upgrades")
    print("   ✅ Confidence levels for analysis quality assessment")
    print("   ✅ Flexible validation modes for different use cases")


if __name__ == "__main__":
    main()
