import pydoc

import mlflow
from pyspark.sql import DataFrame

from main import (
    _resolve_fully_qualified_name,
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


def test_mlflow_integration(spark, mlflow_server):
    """Test that MLflow integration works correctly."""

    def sample_transform(df: DataFrame) -> DataFrame:
        return df.select("*")

    # Start an MLflow run
    with mlflow.start_run():
        # Log the transform function
        log_transform_function(sample_transform, "sample_transform")

        # Verify the function was logged
        run = mlflow.active_run()
        artifacts = mlflow.artifacts.list_artifacts(run.info.run_id)
        assert any(a.path == "transform_code/sample_transform.py" for a in artifacts)
