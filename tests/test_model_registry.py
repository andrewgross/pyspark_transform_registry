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
    create_transform_model,
    infer_pyspark_signature,
)


def test_model_registry_round_trip(spark, mlflow_tracking):
    """Test the complete round-trip using model registry: register, load, and execute."""

    def add_doubled_column(df: DataFrame) -> DataFrame:
        """A simple transform that adds a column with doubled values."""
        return df.withColumn("doubled_value", col("value") * 2)

    # Create test data
    test_data = spark.createDataFrame([(1, "a"), (2, "b"), (3, "c")], ["value", "name"])

    # Start an MLflow run
    with mlflow.start_run() as run:
        transform_name = "add_doubled_column"

        print(f"\nTesting model registry round-trip with run ID: {run.info.run_id}")

        # Step 1: Log the transform function to MLflow model registry
        model_uri = log_transform_function(
            add_doubled_column,
            input_example=test_data,
            registered_model_name=transform_name,
        )
        print(f"✓ Registered transform '{transform_name}' to MLflow model registry")
        print(f"  Model URI: {model_uri}")

        # Step 2: Load the transform function from model registry
        loaded_transform = load_transform_function(transform_name, version="1")
        print(f"✓ Loaded transform '{transform_name}' from model registry")

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

        # Step 6: Test finding transforms in model registry
        found_transforms = find_transform_versions(name=transform_name)
        assert len(found_transforms) > 0
        print(f"✓ Found {len(found_transforms)} transform versions in model registry")

        # Step 7: Test loading by name
        from pyspark_transform_registry import load_transform_function_by_name

        loaded_by_name = load_transform_function_by_name(transform_name)
        by_name_result = loaded_by_name(test_data)
        by_name_rows = by_name_result.collect()

        assert len(by_name_rows) == len(original_rows)
        print("✓ Load by name functionality works correctly")

        print("\n🎉 Complete model registry round-trip test successful!")


def test_model_wrapper_creation(spark):
    """Test creating model wrapper directly."""

    def simple_transform(df: DataFrame) -> DataFrame:
        return df.select("*")

    # Create model wrapper
    model = create_transform_model(simple_transform, "simple_transform")

    # Test wrapper functionality
    assert model.get_function_name() == "simple_transform"
    assert model.get_transform_function() == simple_transform

    # Test predict method
    test_data = spark.createDataFrame([(1, "a"), (2, "b")], ["id", "value"])
    result = model.predict(None, test_data)

    assert result.count() == 2
    assert result.columns == ["id", "value"]

    print("✓ Model wrapper creation and prediction work correctly")


def test_signature_inference(spark):
    """Test signature inference for PySpark transforms."""

    def typed_transform(df: DataFrame) -> DataFrame:
        return df.withColumn("new_col", col("value") + 1)

    # Create test data
    test_data = spark.createDataFrame([(1, "a"), (2, "b")], ["value", "name"])
    output_data = typed_transform(test_data)

    # Test signature inference
    signature = infer_pyspark_signature(typed_transform, test_data, output_data)

    # Basic signature validation
    assert signature is not None
    print("✓ Signature inference works correctly")


def test_backward_compatibility(spark, mlflow_tracking):
    """Test that old API still works for backwards compatibility."""

    def legacy_transform(df: DataFrame) -> DataFrame:
        return df.select("*")

    test_data = spark.createDataFrame([(1, "a")], ["id", "value"])

    with mlflow.start_run():
        # Test legacy validation function
        assert validate_transform_input(legacy_transform, test_data) is True

        # Test resolve function
        fq_name = _resolve_fully_qualified_name(DataFrame)
        assert fq_name == "pyspark.sql.dataframe.DataFrame"
        assert pydoc.locate(fq_name) == DataFrame

        print("✓ Backwards compatibility maintained")
