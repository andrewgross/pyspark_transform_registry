"""Integration tests for versioning functionality with MLflow backend."""

import pytest
import mlflow
from pyspark.sql import DataFrame
from pyspark.sql.functions import col

from pyspark_transform_registry import (
    get_latest_transform_version,
    get_transform_versions, 
    load_transform_function_by_version,
    log_transform_function,
    find_transform_versions,
)


def test_log_transform_with_version(spark, mlflow_tracking):
    """Test logging a transform with explicit version."""
    
    def sample_transform(df: DataFrame) -> DataFrame:
        """A simple transform for testing."""
        return df.select("*")
    
    with mlflow.start_run() as run:
        # Log transform with specific version
        log_transform_function(
            sample_transform, 
            "sample_transform", 
            version="1.2.3"
        )
        
        # Verify the version was stored correctly
        run_data = mlflow.get_run(run.info.run_id)
        assert run_data.data.params["version"] == "1.2.3"
        assert run_data.data.params["transform_name"] == "sample_transform"


def test_log_transform_default_version(spark, mlflow_tracking):
    """Test that default version is applied when none specified."""
    
    def another_transform(df: DataFrame) -> DataFrame:
        return df.select("*")
    
    with mlflow.start_run() as run:
        # Log transform without version (should default to 0.1.0)
        log_transform_function(another_transform, "another_transform")
        
        # Verify default version was applied
        run_data = mlflow.get_run(run.info.run_id)
        assert run_data.data.params["version"] == "0.1.0"


def test_invalid_version_format(spark, mlflow_tracking):
    """Test that invalid version formats are rejected."""
    
    def bad_version_transform(df: DataFrame) -> DataFrame:
        return df.select("*")
    
    with mlflow.start_run():
        # Should raise ValueError for invalid version
        with pytest.raises(ValueError, match="Invalid SemVer format"):
            log_transform_function(
                bad_version_transform,
                "bad_version_transform", 
                version="1.0"  # Missing patch number
            )


def test_duplicate_version_prevention(spark, mlflow_tracking):
    """Test that duplicate name+version combinations are prevented."""
    
    def duplicate_test_transform(df: DataFrame) -> DataFrame:
        return df.select("*")
    
    with mlflow.start_run():
        # Log first version
        log_transform_function(
            duplicate_test_transform,
            "duplicate_test", 
            version="1.0.0"
        )
    
    with mlflow.start_run():
        # Attempt to log same name+version should fail
        with pytest.raises(ValueError, match="already exists"):
            log_transform_function(
                duplicate_test_transform,
                "duplicate_test", 
                version="1.0.0"
            )


def test_allow_version_overwrite(spark, mlflow_tracking):
    """Test that version overwrite can be explicitly allowed."""
    
    def overwrite_transform(df: DataFrame) -> DataFrame:
        return df.select("*")
    
    with mlflow.start_run():
        # Log first version
        log_transform_function(
            overwrite_transform,
            "overwrite_test", 
            version="1.0.0"
        )
    
    with mlflow.start_run():
        # Should succeed with allow_version_overwrite=True
        log_transform_function(
            overwrite_transform,
            "overwrite_test", 
            version="1.0.0",
            allow_version_overwrite=True
        )


def test_find_transforms_by_version(spark, mlflow_tracking):
    """Test searching for transforms by specific version."""
    
    def versioned_transform(df: DataFrame) -> DataFrame:
        return df.withColumn("test_col", col("value") * 2)
    
    with mlflow.start_run():
        log_transform_function(versioned_transform, "versioned_test", version="1.0.0")
    
    with mlflow.start_run():
        log_transform_function(versioned_transform, "versioned_test", version="1.1.0")
    
    # Search for specific version
    v1_runs = find_transform_versions(name="versioned_test", version="1.0.0")
    v11_runs = find_transform_versions(name="versioned_test", version="1.1.0")
    all_runs = find_transform_versions(name="versioned_test")
    
    assert len(v1_runs) == 1
    assert len(v11_runs) == 1
    assert len(all_runs) == 2


def test_get_transform_versions_list(spark, mlflow_tracking):
    """Test getting all versions of a transform."""
    
    def multi_version_transform(df: DataFrame) -> DataFrame:
        return df.select("*")
    
    # Log multiple versions
    versions_to_log = ["1.0.0", "1.1.0", "2.0.0", "1.0.1"]
    
    for version in versions_to_log:
        with mlflow.start_run():
            log_transform_function(
                multi_version_transform, 
                "multi_version_test", 
                version=version
            )
    
    # Get all versions
    found_versions = get_transform_versions("multi_version_test")
    
    # Should be sorted
    expected_sorted = ["1.0.0", "1.0.1", "1.1.0", "2.0.0"]
    assert found_versions == expected_sorted


def test_get_latest_transform_version_function(spark, mlflow_tracking):
    """Test getting the latest version of a transform."""
    
    def latest_test_transform(df: DataFrame) -> DataFrame:
        return df.select("*")
    
    # Log multiple versions out of order
    versions = ["1.0.0", "2.1.0", "1.5.0"]
    
    for version in versions:
        with mlflow.start_run():
            log_transform_function(
                latest_test_transform, 
                "latest_test", 
                version=version
            )
    
    # Get latest version
    latest = get_latest_transform_version("latest_test")
    assert latest == "2.1.0"


def test_get_versions_nonexistent_transform(spark, mlflow_tracking):
    """Test getting versions for a transform that doesn't exist."""
    
    with pytest.raises(ValueError, match="No transform found"):
        get_transform_versions("nonexistent_transform")
    
    # get_latest_transform_version should return None for nonexistent
    result = get_latest_transform_version("nonexistent_transform")
    assert result is None


def test_load_transform_by_version(spark, mlflow_tracking):
    """Test loading a transform by specific version."""
    
    def load_test(df: DataFrame) -> DataFrame:
        """Transform for version loading test."""
        return df.withColumn("loaded_version", col("value") + 100)
    
    # Log different versions
    with mlflow.start_run():
        log_transform_function(load_test, "load_test", version="1.0.0")
    
    with mlflow.start_run():
        log_transform_function(load_test, "load_test", version="2.0.0")
    
    # Load specific version
    loaded_func_v1 = load_transform_function_by_version("load_test", "1.0.0")
    loaded_func_v2 = load_transform_function_by_version("load_test", "2.0.0")
    
    # Both should be callable and work the same (since same function)
    test_data = spark.createDataFrame([(1, "a"), (2, "b")], ["value", "name"])
    
    result_v1 = loaded_func_v1(test_data).collect()
    result_v2 = loaded_func_v2(test_data).collect()
    
    # Results should be identical since it's the same function
    assert len(result_v1) == len(result_v2) == 2
    assert result_v1[0].loaded_version == 101
    assert result_v2[0].loaded_version == 101


def test_load_transform_latest_version(spark, mlflow_tracking):
    """Test loading the latest version of a transform."""
    
    def latest_load_test(df: DataFrame) -> DataFrame:
        return df.withColumn("latest_test", col("value") * 10)
    
    # Log multiple versions
    with mlflow.start_run():
        log_transform_function(latest_load_test, "latest_load_test", version="1.0.0")
    
    with mlflow.start_run():
        log_transform_function(latest_load_test, "latest_load_test", version="1.2.0")
    
    # Load latest version (should get 1.2.0)
    loaded_func = load_transform_function_by_version("latest_load_test", "latest")
    
    # Verify it works
    test_data = spark.createDataFrame([(5,)], ["value"])
    result = loaded_func(test_data).collect()
    assert result[0].latest_test == 50


def test_load_nonexistent_version(spark, mlflow_tracking):
    """Test loading a version that doesn't exist."""
    
    def some_transform(df: DataFrame) -> DataFrame:
        return df.select("*")
    
    with mlflow.start_run():
        log_transform_function(some_transform, "some_test", version="1.0.0")
    
    # Try to load nonexistent version
    with pytest.raises(ValueError, match="No transform found"):
        load_transform_function_by_version("some_test", "2.0.0")
    
    # Try to load from nonexistent transform
    with pytest.raises(ValueError, match="No versions found"):
        load_transform_function_by_version("nonexistent_transform", "latest")