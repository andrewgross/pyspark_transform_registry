"""
Example demonstrating SemVer versioning capabilities in PySpark Transform Registry.

This example shows how to:
1. Log transforms with explicit versions
2. Find and load transforms by version
3. Manage multiple versions of the same transform
4. Use version constraints and latest version features
"""

import mlflow
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, lit

from pyspark_transform_registry import (
    log_transform_function,
    load_transform_function_by_version,
    get_transform_versions,
    get_latest_transform_version,
    find_transform_versions,
    validate_semver,
    increment_version,
)


def create_spark_session():
    """Create a Spark session for the example."""
    return (SparkSession.builder
            .appName("VersioningExample")
            .config("spark.sql.adaptive.enabled", "false")
            .getOrCreate())


def data_cleaner_v1(df: DataFrame) -> DataFrame:
    """
    Initial version of data cleaner - basic cleaning.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Cleaned DataFrame with null values removed
    """
    return df.filter(col("value").isNotNull())


def data_cleaner_v2(df: DataFrame) -> DataFrame:
    """
    Enhanced version of data cleaner - adds validation column.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Cleaned DataFrame with validation status
    """
    return df.filter(col("value").isNotNull()) \
             .withColumn("is_valid", lit(True))


def data_cleaner_v3(df: DataFrame) -> DataFrame:
    """
    Advanced version of data cleaner - comprehensive cleaning.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Thoroughly cleaned and enriched DataFrame
    """
    return df.filter(col("value").isNotNull()) \
             .filter(col("value") > 0) \
             .withColumn("is_valid", lit(True)) \
             .withColumn("cleaned_version", lit("3.0.0"))


def main():
    """Main example function demonstrating versioning features."""
    
    # Initialize Spark and MLflow
    spark = create_spark_session()
    mlflow.set_experiment("versioning_example")
    
    print("=== PySpark Transform Registry Versioning Example ===\n")
    
    # 1. Version Validation
    print("1. Version Validation:")
    valid_versions = ["1.0.0", "2.1.3", "1.0.0-alpha.1"]
    invalid_versions = ["1.0", "1", "v1.0.0"]
    
    for version in valid_versions:
        print(f"   '{version}' is valid: {validate_semver(version)}")
    
    for version in invalid_versions:
        print(f"   '{version}' is valid: {validate_semver(version)}")
    
    print()
    
    # 2. Logging Transforms with Versions
    print("2. Logging Transforms with Versions:")
    
    with mlflow.start_run(run_name="data_cleaner_v1"):
        log_transform_function(
            data_cleaner_v1,
            "data_cleaner",
            version="1.0.0"
        )
        print("   ✓ Logged data_cleaner version 1.0.0")
    
    with mlflow.start_run(run_name="data_cleaner_v2"):
        log_transform_function(
            data_cleaner_v2,
            "data_cleaner", 
            version="2.0.0"
        )
        print("   ✓ Logged data_cleaner version 2.0.0")
    
    with mlflow.start_run(run_name="data_cleaner_v3"):
        log_transform_function(
            data_cleaner_v3,
            "data_cleaner",
            version="3.0.0"
        )
        print("   ✓ Logged data_cleaner version 3.0.0")
    
    print()
    
    # 3. Version Discovery
    print("3. Version Discovery:")
    
    # Get all versions
    versions = get_transform_versions("data_cleaner")
    print(f"   All versions: {versions}")
    
    # Get latest version
    latest_version = get_latest_transform_version("data_cleaner")
    print(f"   Latest version: {latest_version}")
    
    print()
    
    # 4. Loading Specific Versions
    print("4. Loading and Testing Different Versions:")
    
    # Create test data
    test_data = spark.createDataFrame([
        (1, "Alice"),
        (None, "Bob"),  # This will be filtered out
        (-5, "Charlie"),  # This will be handled differently by v3
        (10, "David")
    ], ["value", "name"])
    
    print("   Test data:")
    test_data.show()
    
    # Test version 1.0.0
    cleaner_v1 = load_transform_function_by_version("data_cleaner", "1.0.0")
    result_v1 = cleaner_v1(test_data)
    print("   Results from version 1.0.0:")
    result_v1.show()
    
    # Test version 2.0.0
    cleaner_v2 = load_transform_function_by_version("data_cleaner", "2.0.0")
    result_v2 = cleaner_v2(test_data)
    print("   Results from version 2.0.0:")
    result_v2.show()
    
    # Test version 3.0.0
    cleaner_v3 = load_transform_function_by_version("data_cleaner", "3.0.0")
    result_v3 = cleaner_v3(test_data)
    print("   Results from version 3.0.0:")
    result_v3.show()
    
    # Test latest version
    cleaner_latest = load_transform_function_by_version("data_cleaner", "latest")
    result_latest = cleaner_latest(test_data)
    print("   Results from latest version:")
    result_latest.show()
    
    # 5. Version Management Utilities
    print("5. Version Management Utilities:")
    
    # Version incrementing
    current_version = "1.2.3"
    print(f"   Current version: {current_version}")
    print(f"   Next patch: {increment_version(current_version, 'patch')}")
    print(f"   Next minor: {increment_version(current_version, 'minor')}")
    print(f"   Next major: {increment_version(current_version, 'major')}")
    
    print()
    
    # 6. Advanced Queries
    print("6. Advanced Queries:")
    
    # Find specific version
    specific_runs = find_transform_versions(name="data_cleaner", version="2.0.0")
    print(f"   Runs with version 2.0.0: {len(specific_runs)}")
    
    # Find all versions of data_cleaner
    all_runs = find_transform_versions(name="data_cleaner")
    print(f"   Total data_cleaner runs: {len(all_runs)}")
    
    print()
    
    # 7. Error Handling Examples
    print("7. Error Handling Examples:")
    
    try:
        # Try to log with invalid version
        with mlflow.start_run():
            log_transform_function(data_cleaner_v1, "test", version="1.0")
    except ValueError as e:
        print(f"   ✓ Caught invalid version error: {e}")
    
    try:
        # Try to load nonexistent version
        load_transform_function_by_version("data_cleaner", "99.0.0")
    except ValueError as e:
        print(f"   ✓ Caught nonexistent version error: {e}")
    
    try:
        # Try to get versions for nonexistent transform
        get_transform_versions("nonexistent_transform")
    except ValueError as e:
        print(f"   ✓ Caught nonexistent transform error: {e}")
    
    print("\n=== Example completed successfully! ===")
    
    # Cleanup
    spark.stop()


if __name__ == "__main__":
    main()