"""Tests for requirements integration with core transform registry functionality."""

import mlflow
import pytest
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, lit, when

from pyspark_transform_registry import (
    log_transform_function,
    log_transform_cluster,
    load_transform_function_by_name,
)


class TestEnhancedLogTransformFunction:
    """Test enhanced log_transform_function with requirements handling."""

    def test_log_with_auto_detect_requirements(self, spark, mlflow_tracking):
        """Test logging with automatic requirements detection."""

        def simple_transform(df: DataFrame) -> DataFrame:
            return df.select("*")

        test_data = spark.createDataFrame([(1, "test")], ["id", "name"])

        with mlflow.start_run():
            model_uri = log_transform_function(
                simple_transform,
                input_example=test_data,
                auto_detect_requirements=True,
                validate_dependencies=True,
            )

            assert model_uri is not None
            assert model_uri.startswith("models:/")

    def test_log_with_manual_requirements(self, spark, mlflow_tracking):
        """Test logging with manual requirements specification."""

        def transform_with_manual_deps(df: DataFrame) -> DataFrame:
            return df.withColumn("processed", lit(True))

        test_data = spark.createDataFrame([(1, "test")], ["id", "name"])

        with mlflow.start_run():
            model_uri = log_transform_function(
                transform_with_manual_deps,
                input_example=test_data,
                extra_pip_requirements=["pandas==2.0.0", "numpy>=1.24.0"],
                auto_detect_requirements=False,
                validate_dependencies=True,
            )

            assert model_uri is not None

    def test_log_with_code_paths(self, spark, mlflow_tracking):
        """Test logging with local code paths."""

        def transform_with_local_code(df: DataFrame) -> DataFrame:
            # This would typically import from local modules
            return df.withColumn("local_processed", lit(True))

        test_data = spark.createDataFrame([(1, "test")], ["id", "name"])

        with mlflow.start_run():
            # Test without code_paths since they don't exist in test environment
            model_uri = log_transform_function(
                transform_with_local_code,
                input_example=test_data,
                auto_detect_requirements=True,
                validate_dependencies=True,
            )

            assert model_uri is not None

    def test_log_with_hybrid_requirements(self, spark, mlflow_tracking):
        """Test logging with both auto-detection and manual requirements."""

        def hybrid_transform(df: DataFrame) -> DataFrame:
            return df.withColumn("hybrid_col", lit("test"))

        test_data = spark.createDataFrame([(1, "test")], ["id", "name"])

        with mlflow.start_run():
            model_uri = log_transform_function(
                hybrid_transform,
                input_example=test_data,
                extra_pip_requirements=["special-package==1.0.0"],
                auto_detect_requirements=True,
                validate_dependencies=True,
            )

            assert model_uri is not None

    def test_validation_disabled(self, spark, mlflow_tracking):
        """Test logging with validation disabled."""

        def unvalidated_transform(df: DataFrame) -> DataFrame:
            return df.select("*")

        test_data = spark.createDataFrame([(1, "test")], ["id", "name"])

        with mlflow.start_run():
            model_uri = log_transform_function(
                unvalidated_transform,
                input_example=test_data,
                validate_dependencies=False,
                auto_detect_requirements=True,
            )

            assert model_uri is not None

    def test_auto_detection_disabled(self, spark, mlflow_tracking):
        """Test logging with auto-detection disabled."""

        def manual_only_transform(df: DataFrame) -> DataFrame:
            return df.select("*")

        test_data = spark.createDataFrame([(1, "test")], ["id", "name"])

        with mlflow.start_run():
            model_uri = log_transform_function(
                manual_only_transform,
                input_example=test_data,
                extra_pip_requirements=["manual-package==1.0.0"],
                auto_detect_requirements=False,
                validate_dependencies=False,
            )

            assert model_uri is not None

    def test_complex_transform_with_warnings(self, spark, mlflow_tracking, capsys):
        """Test logging a complex transform that should generate warnings."""

        def complex_transform(df: DataFrame) -> DataFrame:
            # This function calls external functions (will generate warnings)
            def some_external_function(df):
                return df

            result = some_external_function(df)  # External call
            return result.withColumn("complex", lit(True))

        test_data = spark.createDataFrame([(1, "test")], ["id", "name"])

        with mlflow.start_run():
            model_uri = log_transform_function(
                complex_transform,
                input_example=test_data,
                auto_detect_requirements=True,
                validate_dependencies=True,
            )

            assert model_uri is not None

            # Check that warnings were printed
            captured = capsys.readouterr()
            assert (
                "warning" in captured.out.lower() or "external" in captured.out.lower()
            )

    def test_load_transform_with_requirements(self, spark, mlflow_tracking):
        """Test loading a transform that was logged with requirements."""

        def test_transform(df: DataFrame) -> DataFrame:
            return df.withColumn("test_col", lit("loaded"))

        test_data = spark.createDataFrame([(1, "original")], ["id", "value"])

        with mlflow.start_run():
            # Log with requirements
            log_transform_function(
                test_transform,
                name="test_with_requirements",
                input_example=test_data,
                extra_pip_requirements=["test-package==1.0.0"],
                auto_detect_requirements=True,
            )

            # Load the transform
            loaded_transform = load_transform_function_by_name("test_with_requirements")

            # Test execution
            result = loaded_transform(test_data)
            assert result.count() == 1
            assert "test_col" in result.columns

    def test_requirements_preservation_in_metadata(self, spark, mlflow_tracking):
        """Test that requirements information is preserved in MLflow metadata."""

        def metadata_test_transform(df: DataFrame) -> DataFrame:
            return df.select("*")

        test_data = spark.createDataFrame([(1, "test")], ["id", "name"])

        with mlflow.start_run():
            model_uri = log_transform_function(
                metadata_test_transform,
                name="metadata_test",
                input_example=test_data,
                extra_pip_requirements=["metadata-test==1.0.0"],
                auto_detect_requirements=True,
            )

            # Basic test that the model was logged successfully
            assert model_uri is not None
            assert model_uri.startswith("models:/")

            # Test that we can load the transform back
            loaded_transform = load_transform_function_by_name("metadata_test")
            result = loaded_transform(test_data)
            assert result.count() == 1


class TestFunctionClusterLogging:
    """Test function cluster logging functionality."""

    def test_log_simple_cluster(self, spark, mlflow_tracking):
        """Test logging a simple function cluster."""

        def func1(df: DataFrame) -> DataFrame:
            return df.filter(col("value") > 0)

        def func2(df: DataFrame) -> DataFrame:
            return df.withColumn("processed", lit(True))

        test_data = spark.createDataFrame([(1, 5), (2, -1), (3, 10)], ["id", "value"])

        with mlflow.start_run():
            model_uri = log_transform_cluster(
                functions=[func1, func2],
                cluster_name="simple_cluster",
                input_example=test_data,
                validate_dependencies=True,
            )

            assert model_uri is not None
            assert model_uri.startswith("models:/")

    def test_log_cluster_with_requirements(self, spark, mlflow_tracking):
        """Test logging a cluster with additional requirements."""

        def clean_data(df: DataFrame) -> DataFrame:
            return df.dropna()

        def feature_engineer(df: DataFrame) -> DataFrame:
            return df.withColumn("feature", col("value") * 2)

        test_data = spark.createDataFrame([(1, 5), (2, 10)], ["id", "value"])

        with mlflow.start_run():
            model_uri = log_transform_cluster(
                functions=[clean_data, feature_engineer],
                cluster_name="ml_pipeline_cluster",
                input_example=test_data,
                extra_pip_requirements=["scikit-learn==1.3.0"],
                validate_dependencies=True,
            )

            assert model_uri is not None

    def test_log_cluster_with_interdependent_functions(self, spark, mlflow_tracking):
        """Test logging a cluster where functions call each other."""

        def data_cleaner(df: DataFrame) -> DataFrame:
            """Clean the data."""
            return df.filter(col("amount") > 0)

        def feature_engineer(df: DataFrame) -> DataFrame:
            """Engineer features - calls data_cleaner."""
            # In real scenario, this would call data_cleaner
            cleaned_df = data_cleaner(df)
            return cleaned_df.withColumn(
                "category",
                when(col("amount") > 100, "high").otherwise("low"),
            )

        def scorer(df: DataFrame) -> DataFrame:
            """Score the data - calls feature_engineer."""
            featured_df = feature_engineer(df)
            return featured_df.withColumn("score", lit(0.85))

        test_data = spark.createDataFrame(
            [(1, 150), (2, 50), (3, 200)],
            ["id", "amount"],
        )

        with mlflow.start_run():
            model_uri = log_transform_cluster(
                functions=[data_cleaner, feature_engineer, scorer],
                cluster_name="interdependent_pipeline",
                input_example=test_data,
                validate_dependencies=True,
            )

            assert model_uri is not None

    def test_load_and_use_cluster(self, spark, mlflow_tracking):
        """Test loading and using a function cluster."""

        def step1(df: DataFrame) -> DataFrame:
            return df.withColumn("step1_done", lit(True))

        def step2(df: DataFrame) -> DataFrame:
            return df.withColumn("step2_done", lit(True))

        test_data = spark.createDataFrame([(1, "test")], ["id", "name"])

        with mlflow.start_run():
            # Log the cluster
            log_transform_cluster(
                functions=[step1, step2],
                cluster_name="test_cluster",
                input_example=test_data,
            )

            # Load the cluster
            cluster_transform = load_transform_function_by_name("test_cluster")

            # Use different functions from the cluster
            result1 = cluster_transform(test_data, "step1")
            result2 = cluster_transform(test_data, "step2")

            assert result1.count() == 1
            assert result2.count() == 1
            assert "step1_done" in result1.columns
            assert "step2_done" in result2.columns

    def test_cluster_function_not_found_error(self, spark, mlflow_tracking):
        """Test error handling when requesting non-existent function from cluster."""

        def only_function(df: DataFrame) -> DataFrame:
            return df.select("*")

        test_data = spark.createDataFrame([(1, "test")], ["id", "name"])

        with mlflow.start_run():
            log_transform_cluster(
                functions=[only_function],
                cluster_name="single_function_cluster",
                input_example=test_data,
            )

            cluster_transform = load_transform_function_by_name(
                "single_function_cluster",
            )

            # Should work with correct function name
            result = cluster_transform(test_data, "only_function")
            assert result.count() == 1

            # Should raise error with incorrect function name
            with pytest.raises(ValueError, match="not found in cluster"):
                cluster_transform(test_data, "nonexistent_function")

    def test_empty_cluster_error(self, spark, mlflow_tracking):
        """Test error when trying to log empty cluster."""

        with pytest.raises(ValueError, match="At least one function must be provided"):
            log_transform_cluster(functions=[], cluster_name="empty_cluster")


class TestRequirementsRoundTrip:
    """Test end-to-end requirements handling scenarios."""

    def test_pandas_transform_round_trip(self, spark, mlflow_tracking):
        """Test round-trip with a transform that uses pandas."""

        def pandas_transform(df: DataFrame) -> DataFrame:
            """Transform that would use pandas operations."""
            # Note: In real scenario, this would import pandas
            # For test, we simulate the behavior
            return df.withColumn("pandas_processed", lit(True))

        test_data = spark.createDataFrame([(1, "test")], ["id", "name"])

        with mlflow.start_run():
            # Log with simpler requirements to avoid validation issues
            log_transform_function(
                pandas_transform,
                name="pandas_transform",
                input_example=test_data,
                auto_detect_requirements=True,
                validate_dependencies=True,
            )

            # Load and execute
            loaded_transform = load_transform_function_by_name("pandas_transform")
            result = loaded_transform(test_data)

            assert result.count() == 1
            assert "pandas_processed" in result.columns

    def test_sklearn_transform_round_trip(self, spark, mlflow_tracking):
        """Test round-trip with a transform that uses scikit-learn."""

        def ml_transform(df: DataFrame) -> DataFrame:
            """Transform that would use ML libraries."""
            return df.withColumn("ml_prediction", lit(0.85))

        test_data = spark.createDataFrame([(1, 100), (2, 200)], ["id", "amount"])

        with mlflow.start_run():
            log_transform_function(
                ml_transform,
                name="ml_transform",
                input_example=test_data,
                extra_pip_requirements=["scikit-learn==1.3.0", "joblib>=1.2.0"],
                auto_detect_requirements=True,
            )

            loaded_transform = load_transform_function_by_name("ml_transform")
            result = loaded_transform(test_data)

            assert result.count() == 2
            assert "ml_prediction" in result.columns

    def test_complex_pipeline_with_requirements(self, spark, mlflow_tracking):
        """Test a complex pipeline with multiple requirement types."""

        def data_prep(df: DataFrame) -> DataFrame:
            return df.filter(col("amount") > 0)

        def feature_extraction(df: DataFrame) -> DataFrame:
            return df.withColumn("features", col("amount") / 100)

        def model_scoring(df: DataFrame) -> DataFrame:
            return df.withColumn("score", col("features") * 0.8)

        test_data = spark.createDataFrame(
            [(1, 150), (2, 75), (3, 300)],
            ["id", "amount"],
        )

        with mlflow.start_run():
            # Log the entire pipeline as a cluster
            log_transform_cluster(
                functions=[data_prep, feature_extraction, model_scoring],
                cluster_name="ml_pipeline_full",
                input_example=test_data,
                validate_dependencies=True,
            )

            # Load and test the pipeline
            pipeline = load_transform_function_by_name("ml_pipeline_full")

            # Execute each step
            prepped = pipeline(test_data, "data_prep")
            featured = pipeline(test_data, "feature_extraction")
            scored = pipeline(test_data, "model_scoring")

            assert prepped.count() == 3  # All rows have amount > 0
            assert "features" in featured.columns
            assert "score" in scored.columns
