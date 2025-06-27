import pydoc

import mlflow
from pyspark.sql import DataFrame

from pyspark_transform_registry import (
    _resolve_fully_qualified_name,
    find_transform_versions,
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
