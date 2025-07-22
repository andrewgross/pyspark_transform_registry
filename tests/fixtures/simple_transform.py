"""
Simple test fixture for file-based transform registration.
"""

from pyspark.sql import DataFrame
from pyspark.sql.functions import col, lit


def simple_filter(df: DataFrame, min_value: int = 0) -> DataFrame:
    """Filter DataFrame to include only rows where value > min_value."""
    return df.filter(col("value") > min_value)


def add_constant_column(
    df: DataFrame,
    column_name: str = "constant",
    value: int = 42,
) -> DataFrame:
    """Add a constant column to the DataFrame."""
    return df.withColumn(column_name, lit(value))


def chain_transforms(df: DataFrame) -> DataFrame:
    """Chain multiple transforms together."""
    return df.filter(col("value") > 0).withColumn("doubled", col("value") * 2)
