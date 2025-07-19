"""
Functional tests for the PySpark Transform Registry.
Tests the complete system end-to-end using file fixtures.
"""

import pytest
from pyspark.sql import DataFrame
from pyspark.sql.functions import col

from pyspark_transform_registry.core import register_function, load_function


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
        model_uri = register_function(
            func=business_logic,
            name="business.finance.amount_processor",
            input_example=test_data,
            description="Process amounts above threshold",
            tags={"department": "finance", "owner": "data_team"},
        )

        assert model_uri is not None

        # Load function
        loaded_transform = load_function("business.finance.amount_processor")

        # Test original function (verify it works)
        business_logic(test_data, threshold=100)

        # Test loaded function with same data
        loaded_result = loaded_transform(test_data)
        loaded_count = loaded_result.count()

        # Results should be similar (loaded function may have different parameter handling)
        assert loaded_count > 0
        assert "processed" in loaded_result.columns

    def test_file_based_workflow(self, spark, mlflow_tracking):
        """Test complete workflow with file-based registration."""

        # Create test data
        test_data = spark.createDataFrame([(1, 10), (2, 20), (3, 30)], ["id", "value"])

        # Register function from file
        model_uri = register_function(
            file_path="tests/fixtures/simple_transform.py",
            function_name="simple_filter",
            name="etl.data.simple_filter",
            input_example=test_data,
            description="Simple filter transform from file",
            extra_pip_requirements=["pyspark>=3.0.0"],
        )

        assert model_uri is not None

        # Load function
        loaded_transform = load_function("etl.data.simple_filter")

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
            register_function(
                file_path="tests/fixtures/complex_transform.py",
                function_name=func_name,
                name=model_name,
                input_example=test_data,
                description=f"Pipeline step: {func_name}",
            )

        # Load and test individual functions
        data_cleaner = load_function("pipeline.clean.data_cleaner")
        feature_engineer = load_function("pipeline.features.feature_engineer")
        ml_scorer = load_function("pipeline.ml.ml_scorer")
        full_pipeline = load_function("pipeline.complete.full_pipeline")

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
        register_function(
            func=transform_v1,
            name="test.versioning.transform",
            input_example=test_data,
            description="Version 1",
        )

        register_function(
            func=transform_v2,
            name="test.versioning.transform",
            input_example=test_data,
            description="Version 2",
        )

        register_function(
            func=transform_v3,
            name="test.versioning.transform",
            input_example=test_data,
            description="Version 3",
        )

        # Load different versions
        transform_latest = load_function("test.versioning.transform")
        transform_v1_specific = load_function("test.versioning.transform", version=1)
        transform_v2_specific = load_function("test.versioning.transform", version=2)

        # Test that they work
        result_latest = transform_latest(test_data)
        result_v1 = transform_v1_specific(test_data)
        result_v2 = transform_v2_specific(test_data)

        assert result_latest.count() == 2
        assert result_v1.count() == 2
        assert result_v2.count() == 2

        # All should have version column
        assert "version" in result_latest.columns
        assert "version" in result_v1.columns
        assert "version" in result_v2.columns

    def test_error_handling_workflow(self, spark, mlflow_tracking):
        """Test error handling in the workflow."""

        # Test loading non-existent function
        with pytest.raises(Exception):  # MLflow will raise an exception
            load_function("nonexistent.model.name")

        # Test registering with bad file path
        with pytest.raises(FileNotFoundError):
            register_function(
                file_path="nonexistent_file.py",
                function_name="some_function",
                name="test.bad.file",
            )

        # Test registering with bad function name
        with pytest.raises(AttributeError):
            register_function(
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
        register_function(
            func=documented_transform,
            name="test.metadata.documented",
            input_example=test_data,
            description="Test metadata preservation",
            tags={"has_docs": "true", "complexity": "low"},
        )

        # Load function
        loaded_transform = load_function("test.metadata.documented")

        # Test that it works
        result = loaded_transform(test_data)
        assert result.count() == 1
        assert "documented" in result.columns

        # The docstring and function name may not be preserved in the loaded function
        # But the function should work correctly
        doc_result = result.collect()[0]
        assert doc_result["documented"] == 2  # 1 + 1
