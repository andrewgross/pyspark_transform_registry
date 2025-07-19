"""
Complex test fixture with dependencies for file-based transform registration.
"""

from pyspark.sql import DataFrame
from pyspark.sql.functions import col, when, lit, upper


def data_cleaner(df: DataFrame) -> DataFrame:
    """Clean and standardize data."""
    return df.filter(col("amount") > 0).withColumn("status", lit("clean"))


def feature_engineer(df: DataFrame) -> DataFrame:
    """Engineer features from the data."""
    return df.withColumn(
        "risk_category",
        when(col("amount") > 1000, "high")
        .when(col("amount") > 100, "medium")
        .otherwise("low"),
    ).withColumn("amount_upper", upper(col("amount").cast("string")))


def ml_scorer(df: DataFrame) -> DataFrame:
    """Score data using ML logic."""
    # This would normally import ML libraries
    return df.withColumn("score", col("amount") * 0.1)


def full_pipeline(df: DataFrame) -> DataFrame:
    """Full processing pipeline."""
    cleaned = data_cleaner(df)
    featured = feature_engineer(cleaned)
    scored = ml_scorer(featured)
    return scored
