"""
Tests for cross-version compatibility in the PySpark Transform Registry.

This module tests that functions registered in different versions of the registry
can be loaded and used consistently, ensuring backward and forward compatibility
for machine-to-machine operations across system upgrades.
"""

import pytest
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, lit, when

from pyspark_transform_registry import load_function, register_function
from pyspark_transform_registry.schema_constraints import (
    ColumnRequirement,
    PartialSchemaConstraint,
)


class TestVersionedFunctionCompatibility:
    """Test that functions work consistently across versions."""

    def test_simple_function_cross_version_loading(self, spark, mlflow_tracking):
        """Test that simple functions can be loaded across versions."""

        def data_cleaner_v1(df: DataFrame) -> DataFrame:
            """Version 1: Basic data cleaning."""
            return df.filter(col("amount") > 0).select("id", "amount")

        def data_cleaner_v2(df: DataFrame) -> DataFrame:
            """Version 2: Enhanced data cleaning with validation."""
            return (
                df.filter(col("amount") > 0)
                .withColumn("validated", lit(True))
                .select("id", "amount", "validated")
            )

        def data_cleaner_v3(df: DataFrame) -> DataFrame:
            """Version 3: Advanced cleaning with risk scoring."""
            return (
                df.filter(col("amount") > 0)
                .withColumn("validated", lit(True))
                .withColumn(
                    "risk_score",
                    when(col("amount") > 1000, "high")
                    .when(col("amount") > 100, "medium")
                    .otherwise("low"),
                )
                .select("id", "amount", "validated", "risk_score")
            )

        # Register all three versions
        register_function(func=data_cleaner_v1, name="test.version.data_cleaner")
        register_function(func=data_cleaner_v2, name="test.version.data_cleaner")
        register_function(func=data_cleaner_v3, name="test.version.data_cleaner")

        # Test data
        test_df = spark.createDataFrame(
            [(1, 50.0), (2, 150.0), (3, 1500.0)],
            ["id", "amount"],
        )

        # Load and test each version
        v1_func = load_function("test.version.data_cleaner", version=1)
        v2_func = load_function("test.version.data_cleaner", version=2)
        v3_func = load_function("test.version.data_cleaner", version=3)

        # Test v1 behavior
        v1_result = v1_func(test_df)
        assert v1_result.count() == 3
        assert set(v1_result.columns) == {"id", "amount"}

        # Test v2 behavior
        v2_result = v2_func(test_df)
        assert v2_result.count() == 3
        assert set(v2_result.columns) == {"id", "amount", "validated"}
        assert all(row.validated for row in v2_result.collect())

        # Test v3 behavior
        v3_result = v3_func(test_df)
        assert v3_result.count() == 3
        assert set(v3_result.columns) == {"id", "amount", "validated", "risk_score"}

    def test_parameter_evolution_compatibility(self, spark, mlflow_tracking):
        """Test that parameter signatures evolve compatibly across versions."""

        def processor_v1(df: DataFrame) -> DataFrame:
            """Version 1: No parameters."""
            return df.withColumn("processed", lit("v1"))

        def processor_v2(df: DataFrame, version_tag: str = "v2") -> DataFrame:
            """Version 2: Added optional parameter."""
            return df.withColumn("processed", lit(version_tag))

        def processor_v3(
            df: DataFrame,
            version_tag: str = "v3",
            include_metadata: bool = False,
        ) -> DataFrame:
            """Version 3: Added another optional parameter."""
            result = df.withColumn("processed", lit(version_tag))
            if include_metadata:
                result = result.withColumn("metadata", lit("included"))
            return result

        # Register versions
        register_function(func=processor_v1, name="test.version.processor")
        register_function(func=processor_v2, name="test.version.processor")
        register_function(func=processor_v3, name="test.version.processor")

        test_df = spark.createDataFrame([(1, "test")], ["id", "value"])

        # Test that each version works with no parameters
        v1_func = load_function("test.version.processor", version=1)
        v2_func = load_function("test.version.processor", version=2)
        v3_func = load_function("test.version.processor", version=3)

        v1_result = v1_func(test_df)
        v2_result = v2_func(test_df)
        v3_result = v3_func(test_df)

        assert v1_result.collect()[0].processed == "v1"
        assert v2_result.collect()[0].processed == "v2"
        assert v3_result.collect()[0].processed == "v3"

        # Test v2 with parameters
        v2_custom = v2_func(test_df, params={"version_tag": "custom_v2"})
        assert v2_custom.collect()[0].processed == "custom_v2"

        # Test v3 with parameters
        v3_custom = v3_func(
            test_df,
            params={"version_tag": "custom_v3", "include_metadata": True},
        )
        result_row = v3_custom.collect()[0]
        assert result_row.processed == "custom_v3"
        assert result_row.metadata == "included"

    def test_schema_constraint_evolution(self, spark, mlflow_tracking):
        """Test that schema constraints evolve properly across versions."""

        def validator_v1(df: DataFrame) -> DataFrame:
            """Version 1: Requires only id and amount."""
            return df.filter(col("amount") > 0).select("id", "amount")

        def validator_v2(df: DataFrame) -> DataFrame:
            """Version 2: Requires id, amount, and category."""
            return (
                df.filter(col("amount") > 0)
                .filter(col("category").isNotNull())
                .select("id", "amount", "category")
            )

        def validator_v3(df: DataFrame) -> DataFrame:
            """Version 3: Requires id, amount, category, and adds status."""
            return (
                df.filter(col("amount") > 0)
                .filter(col("category").isNotNull())
                .withColumn("status", lit("validated"))
                .select("id", "amount", "category", "status")
            )

        # Create constraints for each version
        v1_constraint = PartialSchemaConstraint(
            required_columns=[
                ColumnRequirement("id", "integer"),
                ColumnRequirement("amount", "double"),
            ],
            added_columns=[],
            preserves_other_columns=False,
        )

        v2_constraint = PartialSchemaConstraint(
            required_columns=[
                ColumnRequirement("id", "integer"),
                ColumnRequirement("amount", "double"),
                ColumnRequirement("category", "string"),
            ],
            added_columns=[],
            preserves_other_columns=False,
        )

        v3_constraint = PartialSchemaConstraint(
            required_columns=[
                ColumnRequirement("id", "integer"),
                ColumnRequirement("amount", "double"),
                ColumnRequirement("category", "string"),
            ],
            added_columns=[ColumnRequirement("status", "string")],
            preserves_other_columns=False,
        )

        # Register with explicit constraints
        register_function(
            func=validator_v1,
            name="test.version.validator",
            schema_constraint=v1_constraint,
        )
        register_function(
            func=validator_v2,
            name="test.version.validator",
            schema_constraint=v2_constraint,
        )
        register_function(
            func=validator_v3,
            name="test.version.validator",
            schema_constraint=v3_constraint,
        )

        # Test data for v1 (minimal requirements)
        v1_compatible_df = spark.createDataFrame(
            [(1, 100.0), (2, 200.0)],
            ["id", "amount"],
        )

        # Test data for v2/v3 (extended requirements)
        v2_compatible_df = spark.createDataFrame(
            [(1, 100.0, "electronics"), (2, 200.0, "books")],
            ["id", "amount", "category"],
        )

        # Load functions with validation
        v1_func = load_function(
            "test.version.validator",
            version=1,
            validate_input=True,
        )
        v2_func = load_function(
            "test.version.validator",
            version=2,
            validate_input=True,
        )
        v3_func = load_function(
            "test.version.validator",
            version=3,
            validate_input=True,
        )

        # Test v1 with minimal data (should work)
        v1_result = v1_func(v1_compatible_df)
        assert v1_result.count() == 2

        # Test v1 with extended data (should work - extra columns ignored)
        v1_extended = v1_func(v2_compatible_df)
        assert v1_extended.count() == 2

        # Test v2 with extended data (should work)
        v2_result = v2_func(v2_compatible_df)
        assert v2_result.count() == 2
        assert "category" in v2_result.columns

        # Test v2 with minimal data (should fail validation due to missing category column)
        try:
            v2_func(v1_compatible_df)
            # If no validation error, it means the schema constraint wasn't applied as expected
            # This could happen if schema inference doesn't detect the category requirement
            # In that case, we expect the function to fail during execution when trying to access category
            assert False, (
                "Expected validation or execution error due to missing category column"
            )
        except ValueError as e:
            # Expected validation error
            assert "Input validation failed" in str(e)
        except Exception as e:
            # Might also get PySpark execution error when trying to access missing column
            assert "category" in str(e) or "UNRESOLVED_COLUMN" in str(e)

        # Test v3 with extended data (should work and add status)
        v3_result = v3_func(v2_compatible_df)
        assert v3_result.count() == 2
        assert "status" in v3_result.columns
        assert all(row.status == "validated" for row in v3_result.collect())


class TestBackwardCompatibility:
    """Test that newer versions maintain backward compatibility."""

    def test_function_signature_backward_compatibility(self, spark, mlflow_tracking):
        """Test that function signatures remain backward compatible."""

        def transform_v1(df: DataFrame) -> DataFrame:
            """Original version."""
            return df.withColumn("version", lit(1))

        def transform_v2(df: DataFrame, multiplier: float = 1.0) -> DataFrame:
            """Enhanced version with optional parameter."""
            return df.withColumn("version", lit(2)).withColumn(
                "multiplied_amount",
                col("amount") * multiplier,
            )

        # Register both versions
        register_function(func=transform_v1, name="test.backward.transform")
        register_function(func=transform_v2, name="test.backward.transform")

        test_df = spark.createDataFrame([(1, 100.0)], ["id", "amount"])

        # Load both versions
        v1_func = load_function("test.backward.transform", version=1)
        v2_func = load_function("test.backward.transform", version=2)

        # Both should work with the same basic input
        v1_result = v1_func(test_df)
        v2_result = v2_func(test_df)  # Using default parameter

        assert v1_result.count() == 1
        assert v2_result.count() == 1
        assert v1_result.collect()[0].version == 1
        assert v2_result.collect()[0].version == 2

        # v2 should also work with explicit parameters
        v2_custom = v2_func(test_df, params={"multiplier": 2.0})
        assert v2_custom.collect()[0].multiplied_amount == 200.0

    def test_schema_backward_compatibility(self, spark, mlflow_tracking):
        """Test that schema requirements remain backward compatible."""

        def basic_filter_v1(df: DataFrame) -> DataFrame:
            """Version 1: Basic filtering."""
            return df.filter(col("amount") > 0)

        def enhanced_filter_v2(df: DataFrame) -> DataFrame:
            """Version 2: Enhanced filtering with optional columns."""
            # Should work with v1 data but can use additional columns if available
            result = df.filter(col("amount") > 0)

            # Use category column if available, otherwise add default
            if "category" in df.columns:
                result = result.filter(col("category").isNotNull())
            else:
                result = result.withColumn("category", lit("default"))

            return result

        # Register both versions
        register_function(func=basic_filter_v1, name="test.backward.filter")
        register_function(func=enhanced_filter_v2, name="test.backward.filter")

        # v1 compatible data (minimal)
        v1_data = spark.createDataFrame(
            [(1, 100.0), (2, -50.0), (3, 200.0)],
            ["id", "amount"],
        )

        # v2 compatible data (with category)
        v2_data = spark.createDataFrame(
            [(1, 100.0, "electronics"), (2, -50.0, None), (3, 200.0, "books")],
            ["id", "amount", "category"],
        )

        # Load both versions
        v1_func = load_function("test.backward.filter", version=1, validate_input=False)
        v2_func = load_function("test.backward.filter", version=2, validate_input=False)

        # Both versions should work with v1 data
        v1_result_on_v1_data = v1_func(v1_data)
        v2_result_on_v1_data = v2_func(v1_data)

        assert v1_result_on_v1_data.count() == 2  # Filters out negative amount
        assert v2_result_on_v1_data.count() == 2  # Should also work
        assert "category" in v2_result_on_v1_data.columns  # Should add default category

        # Both versions should work with v2 data
        v1_result_on_v2_data = v1_func(v2_data)
        v2_result_on_v2_data = v2_func(v2_data)

        assert v1_result_on_v2_data.count() == 2  # Filters out negative amount
        # v2 first filters out negative amount (removing row 2), then filters out null categories
        # After amount > 0 filter: (1, 100.0, "electronics") and (3, 200.0, "books") remain
        # After category isNotNull() filter: both remain since both have non-null categories
        assert (
            v2_result_on_v2_data.count() == 2
        )  # Both remaining rows have valid categories

    def test_api_backward_compatibility(self, spark, mlflow_tracking):
        """Test that API usage patterns remain backward compatible."""

        def api_test_v1(df: DataFrame) -> DataFrame:
            """Version 1: Simple function."""
            return df.select("*")

        def api_test_v2(df: DataFrame, include_extra: bool = False) -> DataFrame:
            """Version 2: Function with optional parameter."""
            if include_extra:
                return df.withColumn("extra", lit("added"))
            return df.select("*")

        # Register both versions
        register_function(func=api_test_v1, name="test.backward.api")
        register_function(func=api_test_v2, name="test.backward.api")

        test_df = spark.createDataFrame([(1, "test")], ["id", "value"])

        # Test that v1 loading and usage patterns still work
        v1_func = load_function("test.backward.api", version=1)
        v1_result = v1_func(test_df)  # Original usage pattern
        assert v1_result.count() == 1

        # Test that v2 works with v1 usage patterns
        v2_func = load_function("test.backward.api", version=2)
        v2_result = v2_func(test_df)  # Should work without params
        assert v2_result.count() == 1
        assert set(v2_result.columns) == {"id", "value"}

        # Test that v2 also supports new patterns
        v2_enhanced = v2_func(test_df, params={"include_extra": True})
        assert "extra" in v2_enhanced.columns


class TestForwardCompatibility:
    """Test that older registry versions can handle newer function features gracefully."""

    def test_schema_constraint_forward_compatibility(self, spark, mlflow_tracking):
        """Test that schema constraints work across different analysis capabilities."""

        def complex_transform(df: DataFrame) -> DataFrame:
            """Function with complex transformations."""
            return (
                df.filter(col("amount") > 0)
                .withColumn(
                    "risk_category",
                    when(col("amount") > 1000, "high")
                    .when(col("amount") > 100, "medium")
                    .otherwise("low"),
                )
                .withColumn("processed_date", lit("2023-01-01"))
                .select("customer_id", "amount", "risk_category", "processed_date")
            )

        # Register with schema inference (simulating newer analysis capabilities)
        register_function(
            func=complex_transform,
            name="test.forward.complex_transform",
            infer_schema=True,
        )

        # Load with different validation modes to test compatibility
        func_strict = load_function(
            "test.forward.complex_transform",
            version=1,
            strict_validation=True,
        )
        func_permissive = load_function(
            "test.forward.complex_transform",
            version=1,
            strict_validation=False,
        )
        func_no_validation = load_function(
            "test.forward.complex_transform",
            version=1,
            validate_input=False,
        )

        test_df = spark.createDataFrame(
            [("cust_1", 150.0), ("cust_2", 1500.0)],
            ["customer_id", "amount"],
        )

        # Strict mode should fail with valid data
        with pytest.raises(ValueError):
            func_strict(test_df)

        # Non Strict modes should work with valid data
        result_permissive = func_permissive(test_df)
        result_no_validation = func_no_validation(test_df)

        for result in [result_permissive, result_no_validation]:
            assert result.count() == 2
            assert "risk_category" in result.columns
            assert "processed_date" in result.columns

    def test_metadata_forward_compatibility(self, spark, mlflow_tracking):
        """Test that metadata extensions don't break older functionality."""

        def metadata_rich_function(
            df: DataFrame,
            processing_mode: str = "standard",
        ) -> DataFrame:
            """Function with rich metadata."""
            return df.withColumn("processing_mode", lit(processing_mode)).withColumn(
                "version",
                lit("2.0"),
            )

        # Register with extensive metadata
        register_function(
            func=metadata_rich_function,
            name="test.forward.metadata_rich",
            description="Function with extensive metadata for forward compatibility testing",
            tags={
                "category": "transformation",
                "complexity": "medium",
                "data_types": "mixed",
                "performance": "optimized",
                "compatibility": "v2+",
            },
            extra_pip_requirements=["numpy>=1.20.0", "pandas>=1.3.0"],
        )

        # Load and verify that the function works regardless of metadata understanding
        loaded_func = load_function("test.forward.metadata_rich", version=1)

        test_df = spark.createDataFrame([(1, "test")], ["id", "value"])

        # Test with defaults
        result_default = loaded_func(test_df)
        assert result_default.collect()[0].processing_mode == "standard"

        # Test with parameters
        result_custom = loaded_func(test_df, params={"processing_mode": "advanced"})
        assert result_custom.collect()[0].processing_mode == "advanced"


class TestVersionConsistency:
    """Test that version handling is consistent across the system."""

    def test_version_numbering_consistency(self, spark, mlflow_tracking):
        """Test that version numbers are assigned consistently."""

        def versioned_func_v1(df: DataFrame) -> DataFrame:
            return df.withColumn("version", lit("1.0"))

        def versioned_func_v2(df: DataFrame) -> DataFrame:
            return df.withColumn("version", lit("2.0"))

        def versioned_func_v3(df: DataFrame) -> DataFrame:
            return df.withColumn("version", lit("3.0"))

        # Register functions in sequence
        register_function(func=versioned_func_v1, name="test.version.consistency")
        register_function(func=versioned_func_v2, name="test.version.consistency")
        register_function(func=versioned_func_v3, name="test.version.consistency")

        test_df = spark.createDataFrame([(1, "test")], ["id", "value"])

        # Test that versions are accessible and consistent
        v1_func = load_function("test.version.consistency", version=1)
        v2_func = load_function("test.version.consistency", version=2)
        v3_func = load_function("test.version.consistency", version=3)

        assert v1_func(test_df).collect()[0].version == "1.0"
        assert v2_func(test_df).collect()[0].version == "2.0"
        assert v3_func(test_df).collect()[0].version == "3.0"

    def test_version_rollback_compatibility(self, spark, mlflow_tracking):
        """Test that rolling back to previous versions works correctly."""

        def stable_function_v1(df: DataFrame) -> DataFrame:
            """Stable version."""
            return df.filter(col("amount") > 0).select("id", "amount")

        def experimental_function_v2(df: DataFrame) -> DataFrame:
            """Experimental version with potential issues."""
            # Simulate a version that might have issues
            return (
                df.filter(col("amount") > 0)
                .withColumn("experimental_feature", lit("beta"))
                .select("id", "amount", "experimental_feature")
            )

        def fixed_function_v3(df: DataFrame) -> DataFrame:
            """Fixed version."""
            return (
                df.filter(col("amount") > 0)
                .withColumn("stable_feature", lit("production"))
                .select("id", "amount", "stable_feature")
            )

        # Register all versions
        register_function(func=stable_function_v1, name="test.version.rollback")
        register_function(func=experimental_function_v2, name="test.version.rollback")
        register_function(func=fixed_function_v3, name="test.version.rollback")

        test_df = spark.createDataFrame(
            [(1, 100.0), (2, -50.0), (3, 200.0)],
            ["id", "amount"],
        )

        # Test that we can always rollback to v1 (stable)
        stable_func = load_function("test.version.rollback", version=1)
        stable_result = stable_func(test_df)

        assert stable_result.count() == 2  # Filters out negative amount
        assert set(stable_result.columns) == {"id", "amount"}

        # Test that v3 works (after fixing v2 issues)
        fixed_func = load_function("test.version.rollback", version=3)
        fixed_result = fixed_func(test_df)

        assert fixed_result.count() == 2
        assert "stable_feature" in fixed_result.columns
        assert all(row.stable_feature == "production" for row in fixed_result.collect())

        # Verify that all versions are still accessible
        experimental_func = load_function("test.version.rollback", version=2)
        experimental_result = experimental_func(test_df)

        assert experimental_result.count() == 2
        assert "experimental_feature" in experimental_result.columns

    def test_concurrent_version_access(self, spark, mlflow_tracking):
        """Test that multiple versions can be accessed concurrently."""

        def concurrent_v1(df: DataFrame) -> DataFrame:
            return df.withColumn("processor", lit("v1"))

        def concurrent_v2(df: DataFrame) -> DataFrame:
            return df.withColumn("processor", lit("v2"))

        # Register versions
        register_function(func=concurrent_v1, name="test.version.concurrent")
        register_function(func=concurrent_v2, name="test.version.concurrent")

        test_df = spark.createDataFrame([(1, "test")], ["id", "value"])

        # Load multiple versions simultaneously
        v1_func = load_function("test.version.concurrent", version=1)
        v2_func = load_function("test.version.concurrent", version=2)

        # Use all versions concurrently
        v1_result = v1_func(test_df)
        v2_result = v2_func(test_df)

        assert v1_result.collect()[0].processor == "v1"
        assert v2_result.collect()[0].processor == "v2"

        # Verify that each function maintains its own state
        assert v1_func != v2_func
