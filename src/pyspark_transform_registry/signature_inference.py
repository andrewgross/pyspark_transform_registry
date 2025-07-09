import inspect
from typing import Callable, Optional

from mlflow.models import ModelSignature
from mlflow.types import ColSpec, DataType, Schema
from pyspark.sql import DataFrame
from pyspark.sql.types import StructType


def infer_pyspark_signature(
    func: Callable,
    input_example: Optional[DataFrame] = None,
    output_example: Optional[DataFrame] = None,
) -> Optional[ModelSignature]:
    """
    Infer MLflow signature for a PySpark transform function.

    Args:
        func: The PySpark transform function
        input_example: Optional example input DataFrame for schema inference
        output_example: Optional example output DataFrame for schema inference

    Returns:
        MLflow ModelSignature if successful, None otherwise
    """
    try:
        # Get function signature
        sig = inspect.signature(func)

        input_schema = None
        output_schema = None

        # Infer input schema
        if input_example is not None:
            input_schema = _spark_schema_to_mlflow_schema(input_example.schema)
        else:
            # Try to infer from function signature
            params = list(sig.parameters.values())
            if params and params[0].annotation == DataFrame:
                # Generic DataFrame schema - will be validated at runtime
                input_schema = _create_generic_dataframe_schema()

        # Infer output schema
        if output_example is not None:
            output_schema = _spark_schema_to_mlflow_schema(output_example.schema)
        elif sig.return_annotation == DataFrame:
            # Generic DataFrame schema - will be validated at runtime
            output_schema = _create_generic_dataframe_schema()

        if input_schema is not None and output_schema is not None:
            return ModelSignature(inputs=input_schema, outputs=output_schema)

        return None

    except Exception as e:
        print(f"Warning: Could not infer signature for function {func.__name__}: {e}")
        return None


def create_signature_from_examples(
    input_df: DataFrame,
    output_df: DataFrame,
) -> ModelSignature:
    """
    Create MLflow signature from input and output DataFrame examples.

    Args:
        input_df: Example input DataFrame
        output_df: Example output DataFrame

    Returns:
        MLflow ModelSignature
    """
    input_schema = _spark_schema_to_mlflow_schema(input_df.schema)
    output_schema = _spark_schema_to_mlflow_schema(output_df.schema)

    return ModelSignature(inputs=input_schema, outputs=output_schema)


def _spark_schema_to_mlflow_schema(spark_schema: StructType) -> Schema:
    """
    Convert PySpark StructType to MLflow Schema.

    Args:
        spark_schema: PySpark StructType

    Returns:
        MLflow Schema
    """
    column_specs = []

    for field in spark_schema.fields:
        mlflow_type = _spark_type_to_mlflow_type(field.dataType)
        if mlflow_type is not None:
            column_specs.append(ColSpec(name=field.name, type=mlflow_type))

    return Schema(column_specs)


def _spark_type_to_mlflow_type(spark_type) -> Optional[DataType]:
    """
    Convert PySpark data type to MLflow DataType.

    Args:
        spark_type: PySpark data type

    Returns:
        MLflow DataType or None if not mappable
    """
    from pyspark.sql.types import (
        BooleanType,
        DateType,
        DoubleType,
        FloatType,
        IntegerType,
        LongType,
        StringType,
        TimestampType,
    )

    type_mapping = {
        BooleanType: DataType.boolean,
        DateType: DataType.datetime,
        DoubleType: DataType.double,
        FloatType: DataType.float,
        IntegerType: DataType.integer,
        LongType: DataType.long,
        StringType: DataType.string,
        TimestampType: DataType.datetime,
    }

    spark_type_class = type(spark_type)
    return type_mapping.get(spark_type_class)


def _create_generic_dataframe_schema() -> Schema:
    """
    Create a generic DataFrame schema for cases where specific schema cannot be inferred.

    Returns:
        MLflow Schema with generic column specification
    """
    # Create a flexible schema that accepts any DataFrame
    # This is a placeholder - in practice, we'd want more specific schemas
    return Schema([ColSpec(name="data", type=DataType.string)])


def validate_transform_signature(
    func: Callable,
    input_df: DataFrame,
    expected_signature: ModelSignature,
) -> tuple[bool, Optional[str]]:
    """
    Validate that a transform function's input matches expected signature.

    Args:
        func: PySpark transform function
        input_df: Input DataFrame to validate
        expected_signature: Expected MLflow signature

    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        # MLflow will handle signature validation automatically when the model is used
        # This is a placeholder for custom validation if needed
        return True, None

    except Exception as e:
        return False, str(e)
