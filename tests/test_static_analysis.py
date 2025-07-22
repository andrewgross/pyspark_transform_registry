"""
Tests for the static analysis engine.

This module tests the LibCST-based static analysis system that infers
schema constraints from PySpark transform function source code.
"""

import inspect

import libcst as cst

from pyspark_transform_registry.schema_constraints import (
    ColumnRequirement,
    ColumnTransformation,
)
from pyspark_transform_registry.static_analysis import analyze_function

# Import the test examples
from pyspark_transform_registry.static_analysis.column_analyzer import (
    find_column_references,
)
from pyspark_transform_registry.static_analysis.operation_analyzer import (
    find_dataframe_operations,
)
from tests.test_data.schema_constraint_examples import add_timestamp, normalize_amounts


class TestLibCSTInfrastructure:
    """Test LibCST analysis infrastructure."""

    def test_libcst_import(self):
        """Test that LibCST is available and working."""
        # Test basic parsing
        source = "df.withColumn('test', df.amount * 2)"
        tree = cst.parse_expression(source)
        assert tree is not None

    def test_function_source_extraction(self):
        """Test extracting source code from functions."""
        source = inspect.getsource(add_timestamp)
        assert "def add_timestamp" in source
        assert "withColumn" in source
        assert "current_timestamp" in source

    def test_parse_function_source(self):
        """Test parsing function source with LibCST."""

        source = inspect.getsource(normalize_amounts)

        tree = cst.parse_module(source)
        assert tree is not None


class TestColumnReferenceDetection:
    """Test detection of column references in function source."""

    def test_detect_dot_notation_column_access(self):
        """Test detecting df.column_name patterns."""
        source = "df.amount * 2"

        result = find_column_references(source)
        assert result["total_references"] == 1
        assert result["unique_columns"] == 1
        assert result["read_columns"] == {"amount"}
        assert result["written_columns"] == set()
        assert result["conditional_columns"] == set()
        assert result["all_detected"] == {"amount"}

    def test_detect_bracket_notation_column_access(self):
        """Test detecting df["column_name"] patterns."""
        source = 'df["revenue"] - df["cost"]'

        result = find_column_references(source)
        assert result["total_references"] == 2
        assert result["unique_columns"] == 2
        assert result["read_columns"] == {"revenue", "cost"}
        assert result["written_columns"] == set()
        assert result["conditional_columns"] == set()
        assert result["all_detected"] == {"revenue", "cost"}

    def test_detect_complex_column_expressions(self):
        """Test detecting columns in complex expressions."""
        source = (
            "F.when(df.category == 'premium', df.amount * 0.9).otherwise(df.amount)"
        )
        result = find_column_references(source)
        assert result["total_references"] == 2
        assert result["unique_columns"] == 2
        assert result["written_columns"] == set()
        assert result["conditional_columns"] == set()
        assert result["all_detected"] == {"category", "amount"}
        assert result["read_columns"] == {"category", "amount"}


class TestDataFrameOperationDetection:
    """Test detection of DataFrame operations."""

    def test_detect_withcolumn_operations(self):
        """Test detecting withColumn operations."""
        source = 'df.withColumn("new_col", F.current_timestamp())'

        result = find_dataframe_operations(source)
        assert len(result["operations"]) == 1
        assert result["operations"][0]["method"] == "withColumn"
        assert result["operations"][0]["affects_rows"] is False
        assert result["operations"][0]["affects_schema"] is True
        assert result["operations"][0]["type"] == "column_transformation"

    def test_detect_chained_operations(self):
        """Test detecting chained DataFrame operations."""
        source = """(
        df.filter(df.amount > 100)
          .withColumn("processed", F.current_timestamp())
          .drop("temp_column")
        )
        """

        result = find_dataframe_operations(source)
        assert len(result["operations"]) == 3

        filter_op = [op for op in result["operations"] if op["method"] == "filter"][0]
        with_column_op = [
            op for op in result["operations"] if op["method"] == "withColumn"
        ][0]
        drop_op = [op for op in result["operations"] if op["method"] == "drop"][0]
        assert filter_op["affects_rows"] is True
        assert filter_op["affects_schema"] is False
        assert filter_op["type"] == "row_filtering"
        assert with_column_op["affects_rows"] is False
        assert with_column_op["affects_schema"] is True
        assert with_column_op["type"] == "column_transformation"
        assert drop_op["affects_rows"] is False
        assert drop_op["affects_schema"] is True
        assert drop_op["type"] == "column_removal"

    def test_detect_filter_operations(self):
        """Test detecting filter operations."""
        source = "df.filter(df.status == 'active')"

        result = find_dataframe_operations(source)

        assert len(result["operations"]) == 1
        assert result["operations"][0]["type"] == "row_filtering"
        assert result["operations"][0]["affects_rows"] is True
        assert result["operations"][0]["affects_schema"] is False

    def test_detect_select_operations(self):
        """Test detecting select operations."""
        source = 'df.select("col1", "col2", "col3")'

        result = find_dataframe_operations(source)

        assert len(result["operations"]) == 1
        assert result["operations"][0]["type"] == "column_selection"
        assert "col1" in result["operations"][0]["args"]
        assert "col2" in result["operations"][0]["args"]
        assert "col3" in result["operations"][0]["args"]
        assert result["operations"][0]["affects_rows"] is False
        assert result["operations"][0]["affects_schema"] is True

    def test_detect_drop_operations(self):
        """Test detecting drop operations."""
        source = 'df.drop("temp_col1", "temp_col2")'

        result = find_dataframe_operations(source)

        assert len(result["operations"]) == 1
        assert result["operations"][0]["type"] == "column_removal"
        assert "temp_col1" in result["operations"][0]["args"]
        assert "temp_col2" in result["operations"][0]["args"]
        assert result["operations"][0]["affects_rows"] is False
        assert result["operations"][0]["affects_schema"] is True


class TestFullFunctionAnalysis:
    """Test full function analysis."""

    def test_analyze_function(self):
        """Test analyzing a full function."""

        result = analyze_function(normalize_amounts)
        assert result.required_columns == [ColumnRequirement("amount", "double")]
        assert result.added_columns == [ColumnTransformation("amount", "add", "double")]
        assert result.removed_columns == []
        assert result.preserves_other_columns
        assert result.warnings == []

    def test_analyze_timestamp_function(self):
        """Test analyzing a full function with timestamp."""

        result = analyze_function(add_timestamp)
        assert result.required_columns == [ColumnRequirement("amount", "double")]
        assert result.added_columns == [
            ColumnTransformation("amount", "add", "timestamp"),
        ]
        assert result.removed_columns == []
        assert result.preserves_other_columns
        assert result.warnings == []
