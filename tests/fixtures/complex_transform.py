"""
Complex test fixture with dependencies for file-based transform registration.
"""

import pyspark.sql.functions as F
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, lit, upper, when


def data_cleaner(df: DataFrame) -> DataFrame:
    """Clean and standardize data."""
    return df.filter(col("amount") > 0).withColumn("status", lit("clean"))


def data_cleaner_f(df: DataFrame) -> DataFrame:
    """Clean and standardize data."""
    return df.filter(F.col("amount") > 0).withColumn("status", F.lit("clean"))


def feature_engineer(df: DataFrame) -> DataFrame:
    """Engineer features from the data."""
    return df.withColumn(
        "risk_category",
        when(col("amount") > 1000, "high")
        .when(col("amount") > 100, "medium")
        .otherwise("low"),
    ).withColumn("amount_upper", upper(col("amount").cast("string")))


def feature_engineer_f(df: DataFrame) -> DataFrame:
    """Engineer features from the data."""
    return df.withColumn(
        "risk_category",
        F.when(F.col("amount") > 1000, "high")
        .when(F.col("amount") > 100, "medium")
        .otherwise("low"),
    ).withColumn("amount_upper", F.upper(F.col("amount").cast("string")))


def ml_scorer(df: DataFrame) -> DataFrame:
    """Score data using ML logic."""
    # This would normally import ML libraries
    return df.withColumn("score", col("amount") * 0.1)


def ml_scorer_f(df: DataFrame) -> DataFrame:
    """Score data using ML logic."""
    # This would normally import ML libraries
    return df.withColumn("score", F.col("amount") * 0.1)


def full_pipeline(df: DataFrame) -> DataFrame:
    """Full processing pipeline."""
    cleaned = data_cleaner(df)
    featured = feature_engineer(cleaned)
    scored = ml_scorer(featured)
    return scored


def full_pipeline_f(df: DataFrame) -> DataFrame:
    """Full processing pipeline."""
    cleaned = data_cleaner(df)
    featured = feature_engineer(cleaned)
    scored = ml_scorer(featured)
    return scored


def custom_dataframe_name(my_df: DataFrame) -> DataFrame:
    """Clean and standardize data."""
    return my_df.filter(F.col("amount") > 0).withColumn("status", F.lit("clean"))
