"""
Test transform examples with expected schema constraints.

This module contains example transform functions and their expected partial
schema constraints for testing the schema inference system.
"""

import pyspark.sql.functions as F
import pytest
from pyspark.sql import DataFrame
from pyspark.sql.functions import current_timestamp

from pyspark_transform_registry.schema_constraints import (
    ColumnRequirement,
    ColumnTransformation,
    PartialSchemaConstraint,
)


# Example 1: Add timestamp column - input schema doesn't matter
def add_timestamp_f(df: DataFrame, *, column_name: str = "created_at") -> DataFrame:
    """Add a timestamp column with current timestamp."""
    return df.withColumn(column_name, F.current_timestamp())


EXPECTED_ADD_TIMESTAMP_F = PartialSchemaConstraint(
    required_columns=[],
    added_columns=[ColumnTransformation("created_at", "add", "timestamp", False)],
    preserves_other_columns=True,
)


def custom_dataframe_name(
    my_df: DataFrame,
    *,
    column_name: str = "created_at",
) -> DataFrame:
    """Add a timestamp column with current timestamp."""
    return my_df.withColumn(column_name, F.current_timestamp())


EXPECTED_CUSTOM_DATAFRAME_NAME = PartialSchemaConstraint(
    required_columns=[],
    added_columns=[ColumnTransformation("created_at", "add", "timestamp", False)],
    preserves_other_columns=True,
)


def no_f_functions(my_df: DataFrame, *, column_name: str = "created_at") -> DataFrame:
    """Add a timestamp column with current timestamp."""
    return my_df.withColumn(column_name, current_timestamp())


EXPECTED_NO_F_FUNCTIONS = PartialSchemaConstraint(
    required_columns=[],
    added_columns=[ColumnTransformation("created_at", "add", "timestamp", False)],
    preserves_other_columns=True,
)


# Example 2: Normalize amounts - requires 'amount' column
def normalize_amounts_f(df: DataFrame, *, scale: float = 1.0) -> DataFrame:
    """Normalize amounts by multiplying by scale factor."""
    return df.withColumn("amount", df.amount * scale)


EXPECTED_NORMALIZE_AMOUNTS_F = PartialSchemaConstraint(
    required_columns=[ColumnRequirement("amount", "double", nullable=True)],
    modified_columns=[ColumnTransformation("amount", "modify", "double")],
    preserves_other_columns=True,
)


# Example 3: Filter with status - requires 'status' column, preserves schema
def filter_active_f(df: DataFrame, *, statuses: list = None) -> DataFrame:
    """Filter DataFrame to only active records."""
    if statuses is None:
        statuses = ["active"]
    return df.filter(df.status.isin(statuses))


EXPECTED_FILTER_ACTIVE_F = PartialSchemaConstraint(
    required_columns=[ColumnRequirement("status", "string", nullable=True)],
    preserves_other_columns=True,
)


# Example 4: Select specific columns - explicit output schema
def select_customer_info_f(df: DataFrame, *, include_phone: bool = False) -> DataFrame:
    """Select customer information columns."""
    cols = ["customer_id", "name", "email"]
    if include_phone:
        cols.append("phone")
    return df.select(*cols)


EXPECTED_SELECT_CUSTOMER_INFO_F = PartialSchemaConstraint(
    required_columns=[
        ColumnRequirement("customer_id", "string", nullable=True),
        ColumnRequirement("name", "string", nullable=True),
        ColumnRequirement("email", "string", nullable=True),
    ],
    optional_columns=[
        ColumnRequirement("phone", "string", nullable=True),
    ],
    preserves_other_columns=False,  # select() doesn't preserve other columns
)


# Example 5: Complex transform with multiple operations
def customer_analytics_f(df: DataFrame, *, min_orders: int = 5) -> DataFrame:
    """Perform customer analytics with multiple transformations."""
    return (
        df.filter(df.order_count >= min_orders)
        .withColumn(
            "vip_status",
            F.when(df.order_count > 50, "VIP").otherwise("Regular"),
        )
        .withColumn("analysis_date", F.current_date())
        .drop("temp_staging_column")
    )


EXPECTED_CUSTOMER_ANALYTICS_F = PartialSchemaConstraint(
    required_columns=[
        ColumnRequirement("order_count", "integer", nullable=True),
        ColumnRequirement("temp_staging_column", "string", nullable=True),
    ],
    added_columns=[
        ColumnTransformation("vip_status", "add", "string", nullable=False),
        ColumnTransformation("analysis_date", "add", "date", nullable=False),
    ],
    removed_columns=["temp_staging_column"],
    preserves_other_columns=True,
)


# Example 6: Function with string operations
def clean_text_data_f(
    df: DataFrame,
    *,
    target_column: str = "description",
) -> DataFrame:
    """Clean text data by trimming and converting to lowercase."""
    return df.withColumn(target_column, F.lower(F.trim(df[target_column])))


EXPECTED_CLEAN_TEXT_DATA_F = PartialSchemaConstraint(
    required_columns=[ColumnRequirement("description", "string", nullable=True)],
    modified_columns=[ColumnTransformation("description", "modify", "string")],
    preserves_other_columns=True,
)


# Example 7: Function with mathematical operations
def calculate_metrics_f(
    df: DataFrame,
    *,
    revenue_col: str = "revenue",
    cost_col: str = "cost",
) -> DataFrame:
    """Calculate profit and margin metrics."""
    return df.withColumn("profit", df[revenue_col] - df[cost_col]).withColumn(
        "margin",
        (df[revenue_col] - df[cost_col]) / df[revenue_col],
    )


EXPECTED_CALCULATE_METRICS_F = PartialSchemaConstraint(
    required_columns=[
        ColumnRequirement("revenue", "double", nullable=True),
        ColumnRequirement("cost", "double", nullable=True),
    ],
    added_columns=[
        ColumnTransformation("profit", "add", "double", nullable=True),
        ColumnTransformation("margin", "add", "double", nullable=True),
    ],
    preserves_other_columns=True,
)


# Example 8: Function with conditional column creation
def add_category_flags_f(
    df: DataFrame,
    *,
    category_col: str = "category",
    flag_prefix: str = "is_",
) -> DataFrame:
    """Add boolean flags for each category."""
    categories = ["premium", "standard", "basic"]
    result_df = df
    for category in categories:
        flag_name = f"{flag_prefix}{category}"
        result_df = result_df.withColumn(
            flag_name,
            F.when(df[category_col] == category, True).otherwise(False),
        )
    return result_df


EXPECTED_ADD_CATEGORY_FLAGS_F = PartialSchemaConstraint(
    required_columns=[ColumnRequirement("category", "string", nullable=True)],
    added_columns=[
        ColumnTransformation("is_premium", "add", "boolean", nullable=False),
        ColumnTransformation("is_standard", "add", "boolean", nullable=False),
        ColumnTransformation("is_basic", "add", "boolean", nullable=False),
    ],
    preserves_other_columns=True,
)


# Example 9: Function with aggregations (challenging for static analysis)
def summarize_by_group_f(
    df: DataFrame,
    *,
    group_col: str = "group_id",
    value_col: str = "value",
) -> DataFrame:
    """Create summary statistics by group."""
    return df.groupBy(group_col).agg(
        F.sum(value_col).alias("total_value"),
        F.avg(value_col).alias("avg_value"),
        F.count(value_col).alias("record_count"),
        F.max(value_col).alias("max_value"),
        F.min(value_col).alias("min_value"),
    )


EXPECTED_SUMMARIZE_BY_GROUP_F = PartialSchemaConstraint(
    required_columns=[
        ColumnRequirement("group_id", "string", nullable=True),
        ColumnRequirement("value", "double", nullable=True),
    ],
    added_columns=[
        ColumnTransformation("total_value", "add", "double", nullable=True),
        ColumnTransformation("avg_value", "add", "double", nullable=True),
        ColumnTransformation("record_count", "add", "integer", nullable=False),
        ColumnTransformation("max_value", "add", "double", nullable=True),
        ColumnTransformation("min_value", "add", "double", nullable=True),
    ],
    preserves_other_columns=False,  # groupBy changes the schema structure
)


# Example 10: Function with UDF (challenging for static analysis)
def apply_business_logic_f(df: DataFrame, *, threshold: float = 100.0) -> DataFrame:
    """Apply complex business logic using UDF."""

    @F.udf("boolean")
    def is_high_value(amount, category):
        if category == "premium":
            return amount > threshold * 0.5
        elif category == "standard":
            return amount > threshold
        else:
            return amount > threshold * 2.0

    return df.withColumn("high_value_flag", is_high_value(df.amount, df.category))


EXPECTED_APPLY_BUSINESS_LOGIC_F = PartialSchemaConstraint(
    required_columns=[
        ColumnRequirement("amount", "double", nullable=True),
        ColumnRequirement("category", "string", nullable=True),
    ],
    added_columns=[
        ColumnTransformation("high_value_flag", "add", "boolean", nullable=True),
    ],
    preserves_other_columns=True,
    warnings=["Contains UDF - static analysis may be incomplete"],
)


# Edge case examples that should be harder to analyze
def dynamic_column_transform(df: DataFrame, *, columns: list = None) -> DataFrame:
    """Transform with dynamic column names - hard to analyze statically."""
    if columns is None:
        columns = ["col1", "col2"]

    result_df = df
    for col in columns:
        result_df = result_df.withColumn(f"processed_{col}", result_df[col] * 2)
    return result_df


EXPECTED_DYNAMIC_COLUMN = PartialSchemaConstraint(
    required_columns=[],  # Can't determine statically
    preserves_other_columns=True,
)


EDGE_CASE_EXAMPLES = [
    (dynamic_column_transform, EXPECTED_DYNAMIC_COLUMN),
]

BASIC_TRANSFORM_EXAMPLES = [
    (add_timestamp_f, EXPECTED_ADD_TIMESTAMP_F),
    (custom_dataframe_name, EXPECTED_CUSTOM_DATAFRAME_NAME),
    (no_f_functions, EXPECTED_NO_F_FUNCTIONS),
    (normalize_amounts_f, EXPECTED_NORMALIZE_AMOUNTS_F),
    (filter_active_f, EXPECTED_FILTER_ACTIVE_F),
    (select_customer_info_f, EXPECTED_SELECT_CUSTOMER_INFO_F),
    (customer_analytics_f, EXPECTED_CUSTOMER_ANALYTICS_F),
    (clean_text_data_f, EXPECTED_CLEAN_TEXT_DATA_F),
    (calculate_metrics_f, EXPECTED_CALCULATE_METRICS_F),
    pytest.param(
        add_category_flags_f,
        EXPECTED_ADD_CATEGORY_FLAGS_F,
        marks=pytest.mark.xfail(reason="Havent figured out how to handle this yet"),
    ),
    (summarize_by_group_f, EXPECTED_SUMMARIZE_BY_GROUP_F),
    (apply_business_logic_f, EXPECTED_APPLY_BUSINESS_LOGIC_F),
]

# Test data for validating constraint examples
ALL_TRANSFORM_EXAMPLES = BASIC_TRANSFORM_EXAMPLES + EDGE_CASE_EXAMPLES
