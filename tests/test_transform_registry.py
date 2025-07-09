import pydoc

import mlflow
from pyspark.sql import DataFrame
from pyspark.sql.functions import col

from pyspark_transform_registry import (
    _resolve_fully_qualified_name,
    find_transform_versions,
    load_transform_function,
    log_transform_function,
    validate_transform_input,
)


def test_resolve_fully_qualified_name():
    """Test that _resolve_fully_qualified_name correctly resolves DataFrame type."""
    fq = _resolve_fully_qualified_name(DataFrame)
    assert fq == "pyspark.sql.dataframe.DataFrame"
    assert pydoc.locate(fq) == DataFrame


def test_validate_transform_input(spark):
    """Test that validate_transform_input correctly validates input types."""

    def sample_func(df: DataFrame) -> DataFrame:
        return df.select("*")

    df = spark.createDataFrame([[1, 2]], ["a", "b"])
    assert validate_transform_input(sample_func, df) is True
    assert validate_transform_input(sample_func, {"a": 1}) is False


def test_mlflow_integration(spark, mlflow_tracking):
    """Test that MLflow integration works correctly."""

    def sample_transform(df: DataFrame) -> DataFrame:
        return df.select("*")

    # Start an MLflow run
    with mlflow.start_run() as run:
        print(f"\nMLflow tracking URI: {mlflow.get_tracking_uri()}")
        print(f"MLflow experiment ID: {run.info.experiment_id}")
        print(f"MLflow run ID: {run.info.run_id}")

        # Log the transform function
        log_transform_function(sample_transform, "sample_transform")

        # Verify the function was logged
        # First, check that the 'transform_code' directory exists at the root
        root_artifacts = mlflow.artifacts.list_artifacts(run_id=run.info.run_id)
        print(f"\nRoot artifacts found: {[a.path for a in root_artifacts]}")
        assert any(a.path == "transform_code" for a in root_artifacts)

        # Now, list artifacts in the 'transform_code' directory
        code_artifacts = mlflow.artifacts.list_artifacts(
            run_id=run.info.run_id,
            artifact_path="transform_code",
        )
        print(f"Artifacts in 'transform_code': {[a.path for a in code_artifacts]}")
        assert any(
            a.path == "transform_code/sample_transform.py" for a in code_artifacts
        )

        # Verify metadata was logged correctly
        run_data = mlflow.get_run(run.info.run_id)
        print(f"\nRun data tags: {run_data.data.tags}")
        print(f"Run data params: {run_data.data.params}")
        print(f"Run data metrics: {run_data.data.metrics}")

        # Test finding transform versions
        transforms = find_transform_versions(name="sample_transform")
        print(f"\nFound transforms: {transforms}")
        assert len(transforms) > 0


def test_mlflow_round_trip_execution(spark, mlflow_tracking):
    """Test the complete round-trip: save transform to MLflow, download it, and execute it."""

    def add_doubled_column(df: DataFrame) -> DataFrame:
        """A simple transform that adds a column with doubled values."""
        return df.withColumn("doubled_value", col("value") * 2)

    # Create test data
    test_data = spark.createDataFrame([(1, "a"), (2, "b"), (3, "c")], ["value", "name"])

    # Start an MLflow run
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        transform_name = "add_doubled_column"

        print(f"\nTesting round-trip execution with run ID: {run_id}")

        # Step 1: Log the transform function to MLflow
        log_transform_function(add_doubled_column, transform_name)
        print(f"✓ Logged transform '{transform_name}' to MLflow")

        # Step 2: Load the transform function back from MLflow
        loaded_transform = load_transform_function(run_id, transform_name)
        print(f"✓ Loaded transform '{transform_name}' from MLflow")

        # Step 3: Execute both the original and loaded functions
        original_result = add_doubled_column(test_data)
        loaded_result = loaded_transform(test_data)

        # Step 4: Convert to pandas for easy comparison (collect small datasets)
        original_rows = original_result.collect()
        loaded_rows = loaded_result.collect()

        print(f"Original result: {original_rows}")
        print(f"Loaded result: {loaded_rows}")

        # Step 5: Verify the results are identical
        assert len(original_rows) == len(loaded_rows)

        # Sort both results to ensure consistent comparison
        original_sorted = sorted(original_rows, key=lambda x: x.value)
        loaded_sorted = sorted(loaded_rows, key=lambda x: x.value)

        for orig_row, loaded_row in zip(original_sorted, loaded_sorted):
            assert orig_row.value == loaded_row.value
            assert orig_row.name == loaded_row.name
            assert orig_row.doubled_value == loaded_row.doubled_value
            # Verify the transformation logic worked correctly
            assert orig_row.doubled_value == orig_row.value * 2

        print("✓ Both original and loaded transforms produced identical results")

        # Step 6: Test that the loaded function has the correct signature and metadata
        import inspect

        original_sig = inspect.signature(add_doubled_column)
        loaded_sig = inspect.signature(loaded_transform)

        assert str(original_sig) == str(loaded_sig)
        print(f"✓ Function signatures match: {original_sig}")

        # Step 7: Verify the function can be found using search
        found_transforms = find_transform_versions(name=transform_name)
        assert len(found_transforms) > 0
        print("✓ Transform can be found via search functionality")

        print("\n🎉 Complete round-trip test successful!")
