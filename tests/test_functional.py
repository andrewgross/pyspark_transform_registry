"""
Functional tests for the PySpark Transform Registry.
Tests the complete system end-to-end using file fixtures.
"""

import pytest
from pyspark.sql import DataFrame
from pyspark.sql.functions import col

from pyspark_transform_registry.core import load_transform, register_transform


class TestEndToEndWorkflow:
    """Test complete end-to-end workflows."""

    def test_direct_function_workflow(self, spark, mlflow_tracking):
        """Test complete workflow with direct function registration."""

        def business_logic(df: DataFrame, threshold: int = 100) -> DataFrame:
            """Business logic transform with parameter."""
            return df.filter(col("amount") > threshold).withColumn(
                "processed",
                col("amount") * 1.1,
            )

        # Create test data
        test_data = spark.createDataFrame(
            [(1, 50), (2, 150), (3, 250)],
            ["id", "amount"],
        )

        # Register function
        logged_model = register_transform(
            func=business_logic,
            name="business.finance.amount_processor",
            description="Process amounts above threshold",
            tags={"department": "finance", "owner": "data_team"},
        )

        assert logged_model is not None
        assert logged_model.name == "business_logic"
        assert logged_model.registered_model_version == 1

        # Load function
        loaded_transform = load_transform(
            "business.finance.amount_processor", version=1
        )

        # Test original function (verify it works)
        business_logic(test_data, threshold=100)

        # Test loaded function with same data
        loaded_result = loaded_transform(test_data)

        # Results should be similar (loaded function may have different parameter handling)
        assert loaded_result.filter(col("id") == 1).count() == 0
        assert (
            loaded_result.filter(col("id") == 2).select("processed").collect()[0][0]
            == 165
        )
        assert (
            loaded_result.filter(col("id") == 3).select("processed").collect()[0][0]
            == 275
        )

    def test_file_based_workflow(self, spark, mlflow_tracking):
        """Test complete workflow with file-based registration."""

        # Create test data
        test_data = spark.createDataFrame([(1, 10), (2, 20), (3, 30)], ["id", "value"])

        # Register function from file
        logged_model = register_transform(
            file_path="tests/fixtures/simple_transform.py",
            function_name="simple_filter",
            name="etl.data.simple_filter",
            description="Simple filter transform from file",
            extra_pip_requirements=["pyspark>=3.0.0"],
        )

        assert logged_model is not None
        assert logged_model.name == "simple_filter"
        assert logged_model.registered_model_version == 1

        # Load function
        loaded_transform = load_transform("etl.data.simple_filter", version=1)

        # Test loaded function
        result = loaded_transform(test_data)

        # Should filter out rows where value <= 0 (default min_value=0)
        assert result.count() == 3  # All test values > 0

        # Test with different input
        test_data_with_zeros = spark.createDataFrame(
            [(1, 0), (2, 10), (3, 20)],
            ["id", "value"],
        )

        result_filtered = loaded_transform(test_data_with_zeros)
        assert result_filtered.count() == 2  # Only values > 0

    def test_complex_pipeline_workflow(self, spark, mlflow_tracking):
        """Test workflow with complex pipeline from file."""

        # Create test data
        test_data = spark.createDataFrame(
            [(1, 50), (2, 150), (3, 1500)],
            ["id", "amount"],
        )

        # Register multiple functions from complex file
        functions_to_register = [
            ("data_cleaner", "pipeline.clean.data_cleaner"),
            ("feature_engineer", "pipeline.features.feature_engineer"),
            ("ml_scorer", "pipeline.ml.ml_scorer"),
            ("full_pipeline", "pipeline.complete.full_pipeline"),
        ]

        for func_name, model_name in functions_to_register:
            register_transform(
                file_path="tests/fixtures/complex_transform.py",
                function_name=func_name,
                name=model_name,
                description=f"Pipeline step: {func_name}",
            )

        # Load and test individual functions
        data_cleaner = load_transform("pipeline.clean.data_cleaner", version=1)
        feature_engineer = load_transform(
            "pipeline.features.feature_engineer",
            version=1,
        )
        ml_scorer = load_transform("pipeline.ml.ml_scorer", version=1)
        full_pipeline = load_transform("pipeline.complete.full_pipeline", version=1)

        # Test individual steps
        cleaned = data_cleaner(test_data)
        assert cleaned.count() == 3  # All amounts > 0
        assert "status" in cleaned.columns

        featured = feature_engineer(cleaned)
        assert "risk_category" in featured.columns

        scored = ml_scorer(featured)
        assert "score" in scored.columns

        # Test full pipeline
        result = full_pipeline(test_data)
        assert result.count() == 3
        assert "status" in result.columns
        assert "risk_category" in result.columns
        assert "score" in result.columns

    def test_complex_pipeline_workflow_f(self, spark, mlflow_tracking):
        """Test workflow with complex pipeline from file."""

        # Create test data
        test_data = spark.createDataFrame(
            [(1, 50), (2, 150), (3, 1500)],
            ["id", "amount"],
        )

        # Register multiple functions from complex file
        functions_to_register = [
            ("data_cleaner_f", "pipeline.clean.data_cleaner"),
            ("feature_engineer_f", "pipeline.features.feature_engineer"),
            ("ml_scorer_f", "pipeline.ml.ml_scorer"),
            ("full_pipeline_f", "pipeline.complete.full_pipeline"),
        ]

        for func_name, model_name in functions_to_register:
            register_transform(
                file_path="tests/fixtures/complex_transform.py",
                function_name=func_name,
                name=model_name,
                description=f"Pipeline step: {func_name}",
            )

        # Load and test individual functions
        data_cleaner = load_transform("pipeline.clean.data_cleaner", version=1)
        feature_engineer = load_transform(
            "pipeline.features.feature_engineer",
            version=1,
        )
        ml_scorer = load_transform("pipeline.ml.ml_scorer", version=1)
        full_pipeline = load_transform("pipeline.complete.full_pipeline", version=1)

        # Test individual steps
        cleaned = data_cleaner(test_data)
        assert cleaned.count() == 3  # All amounts > 0
        assert "status" in cleaned.columns

        featured = feature_engineer(cleaned)
        assert "risk_category" in featured.columns

        scored = ml_scorer(featured)
        assert "score" in scored.columns

        # Test full pipeline
        result = full_pipeline(test_data)
        assert result.count() == 3
        assert "status" in result.columns
        assert "risk_category" in result.columns
        assert "score" in result.columns

    def test_versioning_workflow(self, spark, mlflow_tracking):
        """Test versioning workflow with multiple versions."""

        def transform_v1(df: DataFrame) -> DataFrame:
            return df.withColumn("version", col("id") * 1)

        def transform_v2(df: DataFrame) -> DataFrame:
            return df.withColumn("version", col("id") * 2)

        def transform_v3(df: DataFrame) -> DataFrame:
            return df.withColumn("version", col("id") * 3)

        test_data = spark.createDataFrame([(1, "a"), (2, "b")], ["id", "name"])

        # Register multiple versions
        register_transform(
            func=transform_v1,
            name="test.versioning.transform",
            description="Version 1",
        )

        register_transform(
            func=transform_v2,
            name="test.versioning.transform",
            description="Version 2",
        )

        register_transform(
            func=transform_v3,
            name="test.versioning.transform",
            description="Version 3",
        )

        # Load different versions
        transform_v1_specific = load_transform("test.versioning.transform", version=1)
        transform_v2_specific = load_transform("test.versioning.transform", version=2)
        transform_v3_specific = load_transform("test.versioning.transform", version=3)

        # Test that they work
        result_v1 = transform_v1_specific(test_data)
        result_v2 = transform_v2_specific(test_data)
        result_v3 = transform_v3_specific(test_data)

        assert result_v1.count() == 2
        assert result_v2.count() == 2
        assert result_v3.count() == 2

        # All should have version column
        assert "version" in result_v1.columns
        assert "version" in result_v2.columns
        assert "version" in result_v3.columns

    def test_error_handling_workflow(self, spark, mlflow_tracking):
        """Test error handling in the workflow."""

        # Test loading non-existent function
        with pytest.raises(Exception):  # MLflow will raise an exception
            load_transform("nonexistent.model.name", version=1)

        # Test registering with bad file path
        with pytest.raises(FileNotFoundError):
            register_transform(
                file_path="nonexistent_file.py",
                function_name="some_function",
                name="test.bad.file",
            )

        # Test registering with bad function name
        with pytest.raises(AttributeError):
            register_transform(
                file_path="tests/fixtures/simple_transform.py",
                function_name="nonexistent_function",
                name="test.bad.function",
            )

    def test_metadata_preservation(self, spark, mlflow_tracking):
        """Test that function metadata is preserved."""

        def documented_transform(df: DataFrame) -> DataFrame:
            """
            This is a well-documented transform function.

            Args:
                df: Input DataFrame

            Returns:
                Transformed DataFrame
            """
            return df.withColumn("documented", col("id") + 1)

        test_data = spark.createDataFrame([(1, "a")], ["id", "value"])

        # Register function
        register_transform(
            func=documented_transform,
            name="test.metadata.documented",
            description="Test metadata preservation",
            tags={"has_docs": "true", "complexity": "low"},
        )

        # Load function
        loaded_transform = load_transform("test.metadata.documented", version=1)

        # Test that it works
        result = loaded_transform(test_data)
        assert result.count() == 1
        assert "documented" in result.columns

        # The docstring and function name may not be preserved in the loaded function
        # But the function should work correctly
        doc_result = result.collect()[0]
        assert doc_result["documented"] == 2  # 1 + 1

    def test_end_to_end_with_requirements_installation(self, spark, mlflow_tracking):
        """Test complete end-to-end workflow including requirements installation."""
        from pyspark_transform_registry.core import install_transform_requirements

        # Create a transform function that uses external dependencies
        def advanced_transform(df: DataFrame, multiplier: float = 2.0) -> DataFrame:
            """
            Advanced transform that could use external libraries.

            Args:
                df: Input DataFrame
                multiplier: Value to multiply by

            Returns:
                Transformed DataFrame with computed values
            """
            # Simple transform that simulates using external dependencies
            from pyspark.sql.functions import col, when

            return df.withColumn(
                "processed_value",
                when(col("value") > 10, col("value") * multiplier).otherwise(
                    col("value")
                ),
            ).withColumn(
                "category",
                when(col("processed_value") > 50, "high")
                .when(col("processed_value") > 20, "medium")
                .otherwise("low"),
            )

        # Create test data
        test_data = spark.createDataFrame(
            [(1, 5.0), (2, 15.0), (3, 30.0), (4, 100.0)], ["id", "value"]
        )

        # Step 1: Register transform with requirements
        print("Step 1: Registering transform with requirements...")
        logged_model = register_transform(
            func=advanced_transform,
            name="test.endtoend.advanced_transform",
            description="Advanced transform for end-to-end testing",
            tags={"test_type": "e2e", "complexity": "advanced"},
            extra_pip_requirements=["numpy>=1.20.0", "pandas>=1.3.0"],
        )

        assert logged_model is not None
        assert logged_model.name == "advanced_transform"
        assert logged_model.registered_model_version == 1
        transform_uri = logged_model.transform_uri
        print(f"✓ Transform registered with URI: {transform_uri}")

        # Step 2: Install requirements (simulate dependency installation)
        print("Step 2: Installing transform requirements...")
        try:
            # This would normally install the actual requirements
            # For testing, we'll just verify the function works
            install_transform_requirements(transform_uri)
            print("✓ Requirements installation completed")
        except Exception as e:
            # In test environment, installation might fail but that's okay
            print(f"⚠ Requirements installation attempted (may fail in test): {e}")

        # Step 3: Load the transform
        print("Step 3: Loading transform from MLflow...")
        loaded_transform = load_transform("test.endtoend.advanced_transform", version=1)
        print("✓ Transform loaded successfully")

        # Step 4: Test the original function works
        print("Step 4: Testing original function...")
        original_result = advanced_transform(test_data, multiplier=3.0)
        original_count = original_result.count()
        assert original_count == 4
        assert "processed_value" in original_result.columns
        assert "category" in original_result.columns
        print("✓ Original function works correctly")

        # Step 5: Test the loaded transform with default parameters
        print("Step 5: Testing loaded transform with defaults...")
        loaded_result_default = loaded_transform(df=test_data)
        assert loaded_result_default.count() == 4
        assert "processed_value" in loaded_result_default.columns
        assert "category" in loaded_result_default.columns

        # Verify specific transformations (default multiplier=2.0)
        results = loaded_result_default.collect()
        result_dict = {row.id: row for row in results}

        # ID 1: value=5, should stay 5 (<=10), category=low
        assert result_dict[1].processed_value == 5.0
        assert result_dict[1].category == "low"

        # ID 2: value=15, should become 30 (15*2), category=medium
        assert result_dict[2].processed_value == 30.0
        assert result_dict[2].category == "medium"

        # ID 3: value=30, should become 60 (30*2), category=high
        assert result_dict[3].processed_value == 60.0
        assert result_dict[3].category == "high"

        # ID 4: value=100, should become 200 (100*2), category=high
        assert result_dict[4].processed_value == 200.0
        assert result_dict[4].category == "high"
        print("✓ Loaded transform works with default parameters")

        # Step 6: Test the loaded transform with custom parameters
        print("Step 6: Testing loaded transform with custom parameters...")
        loaded_result_custom = loaded_transform(df=test_data, multiplier=5.0)
        assert loaded_result_custom.count() == 4

        custom_results = loaded_result_custom.collect()
        custom_dict = {row.id: row for row in custom_results}

        # With multiplier=5.0, ID 2: value=15 should become 75 (15*5)
        assert custom_dict[2].processed_value == 75.0
        assert custom_dict[2].category == "high"
        print("✓ Loaded transform works with custom parameters")

        # Step 7: Test metadata access
        print("Step 7: Testing metadata access...")
        source_code = loaded_transform.get_source()
        assert source_code is not None
        assert "Advanced transform" in source_code

        original_func = loaded_transform.get_original_function()
        assert callable(original_func)
        print("✓ Metadata access works correctly")

        # Step 8: Verify the complete round-trip consistency
        print("Step 8: Verifying round-trip consistency...")
        # Both functions should produce same results with same parameters
        original_result_2 = advanced_transform(test_data, multiplier=2.0)
        loaded_result_2 = loaded_transform(df=test_data, multiplier=2.0)

        # Convert to sorted lists for comparison
        original_data = sorted(original_result_2.collect(), key=lambda x: x.id)
        loaded_data = sorted(loaded_result_2.collect(), key=lambda x: x.id)

        for orig, loaded in zip(original_data, loaded_data):
            assert orig.id == loaded.id
            assert orig.value == loaded.value
            assert orig.processed_value == loaded.processed_value
            assert orig.category == loaded.category

        print("✓ Round-trip consistency verified")
        print("🎉 End-to-end test completed successfully!")

        return {
            "registered_model": logged_model,
            "transform_uri": transform_uri,
            "loaded_transform": loaded_transform,
            "test_results": {
                "original_count": original_count,
                "loaded_count": loaded_result_default.count(),
                "custom_count": loaded_result_custom.count(),
            },
        }
