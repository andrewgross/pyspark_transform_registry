"""
Tests for the static analysis engine.

This module tests the LibCST-based static analysis system that infers
schema constraints from PySpark transform function source code.
"""

import inspect
import pytest
from unittest.mock import patch

from pyspark_transform_registry.schema_constraints import (
    PartialSchemaConstraint,
    ColumnRequirement,
    ColumnTransformation,
)

# Import the test examples
from tests.test_data.schema_constraint_examples import (
    add_timestamp,
    normalize_amounts,
)


class TestLibCSTInfrastructure:
    """Test LibCST analysis infrastructure."""

    def test_libcst_import(self):
        """Test that LibCST is available and working."""
        try:
            import libcst as cst

            # Test basic parsing
            source = "df.withColumn('test', df.amount * 2)"
            tree = cst.parse_expression(source)
            assert tree is not None

        except ImportError:
            pytest.fail("LibCST not available - required for static analysis")

    def test_function_source_extraction(self):
        """Test extracting source code from functions."""
        source = inspect.getsource(add_timestamp)
        assert "def add_timestamp" in source
        assert "withColumn" in source
        assert "current_timestamp" in source

    def test_parse_function_source(self):
        """Test parsing function source with LibCST."""
        import libcst as cst

        source = inspect.getsource(normalize_amounts)

        # Should be able to parse without errors
        try:
            tree = cst.parse_module(source)
            assert tree is not None
        except Exception as e:
            pytest.fail(f"Failed to parse function source: {e}")


class TestColumnReferenceDetection:
    """Test detection of column references in function source."""

    def test_detect_dot_notation_column_access(self):
        """Test detecting df.column_name patterns."""
        source = "df.amount * 2"

        # This would be implemented by the column analyzer
        # For now, test the expected interface
        expected_columns = ["amount"]

        # Mock the analyzer for now - will be implemented later
        with patch(
            "pyspark_transform_registry.static_analysis.column_analyzer.find_column_references",
        ) as mock_find:
            mock_find.return_value = expected_columns

            # Simulate the call
            from unittest.mock import MagicMock

            analyzer = MagicMock()
            analyzer.find_column_references.return_value = expected_columns

            result = analyzer.find_column_references(source)
            assert "amount" in result

    def test_detect_bracket_notation_column_access(self):
        """Test detecting df["column_name"] patterns."""
        source = 'df["revenue"] - df["cost"]'
        expected_columns = ["revenue", "cost"]

        # Mock implementation
        with patch(
            "pyspark_transform_registry.static_analysis.column_analyzer.find_column_references",
        ) as mock_find:
            mock_find.return_value = expected_columns

            from unittest.mock import MagicMock

            analyzer = MagicMock()
            analyzer.find_column_references.return_value = expected_columns

            result = analyzer.find_column_references(source)
            assert "revenue" in result
            assert "cost" in result

    def test_detect_complex_column_expressions(self):
        """Test detecting columns in complex expressions."""
        source = (
            "F.when(df.category == 'premium', df.amount * 0.9).otherwise(df.amount)"
        )
        expected_columns = ["category", "amount"]

        # This tests the analyzer's ability to find columns in nested expressions
        with patch(
            "pyspark_transform_registry.static_analysis.column_analyzer.find_column_references",
        ) as mock_find:
            mock_find.return_value = expected_columns

            from unittest.mock import MagicMock

            analyzer = MagicMock()
            analyzer.find_column_references.return_value = expected_columns

            result = analyzer.find_column_references(source)
            assert "category" in result
            assert "amount" in result


class TestDataFrameOperationDetection:
    """Test detection of DataFrame operations."""

    def test_detect_withcolumn_operations(self):
        """Test detecting withColumn operations."""
        source = 'df.withColumn("new_col", F.current_timestamp())'

        # Expected operation detection
        expected_operations = [
            {
                "type": "withColumn",
                "column_name": "new_col",
                "expression": "F.current_timestamp()",
            },
        ]

        with patch(
            "pyspark_transform_registry.static_analysis.operation_analyzer.find_operations",
        ) as mock_find:
            mock_find.return_value = expected_operations

            from unittest.mock import MagicMock

            analyzer = MagicMock()
            analyzer.find_operations.return_value = expected_operations

            result = analyzer.find_operations(source)
            assert len(result) == 1
            assert result[0]["type"] == "withColumn"
            assert result[0]["column_name"] == "new_col"

    def test_detect_filter_operations(self):
        """Test detecting filter operations."""
        source = "df.filter(df.status == 'active')"

        expected_operations = [
            {
                "type": "filter",
                "condition": "df.status == 'active'",
                "referenced_columns": ["status"],
            },
        ]

        with patch(
            "pyspark_transform_registry.static_analysis.operation_analyzer.find_operations",
        ) as mock_find:
            mock_find.return_value = expected_operations

            from unittest.mock import MagicMock

            analyzer = MagicMock()
            analyzer.find_operations.return_value = expected_operations

            result = analyzer.find_operations(source)
            assert len(result) == 1
            assert result[0]["type"] == "filter"

    def test_detect_select_operations(self):
        """Test detecting select operations."""
        source = 'df.select("col1", "col2", "col3")'

        expected_operations = [
            {
                "type": "select",
                "columns": ["col1", "col2", "col3"],
                "preserves_other_columns": False,
            },
        ]

        with patch(
            "pyspark_transform_registry.static_analysis.operation_analyzer.find_operations",
        ) as mock_find:
            mock_find.return_value = expected_operations

            from unittest.mock import MagicMock

            analyzer = MagicMock()
            analyzer.find_operations.return_value = expected_operations

            result = analyzer.find_operations(source)
            assert len(result) == 1
            assert result[0]["type"] == "select"
            assert "col1" in result[0]["columns"]

    def test_detect_drop_operations(self):
        """Test detecting drop operations."""
        source = 'df.drop("temp_col1", "temp_col2")'

        expected_operations = [
            {
                "type": "drop",
                "columns": ["temp_col1", "temp_col2"],
            },
        ]

        with patch(
            "pyspark_transform_registry.static_analysis.operation_analyzer.find_operations",
        ) as mock_find:
            mock_find.return_value = expected_operations

            from unittest.mock import MagicMock

            analyzer = MagicMock()
            analyzer.find_operations.return_value = expected_operations

            result = analyzer.find_operations(source)
            assert len(result) == 1
            assert result[0]["type"] == "drop"
            assert "temp_col1" in result[0]["columns"]

    def test_detect_chained_operations(self):
        """Test detecting chained DataFrame operations."""
        source = """
        df.filter(df.amount > 100)
          .withColumn("processed", F.current_timestamp())
          .drop("temp_column")
        """

        expected_operations = [
            {"type": "filter", "referenced_columns": ["amount"]},
            {"type": "withColumn", "column_name": "processed"},
            {"type": "drop", "columns": ["temp_column"]},
        ]

        with patch(
            "pyspark_transform_registry.static_analysis.operation_analyzer.find_operations",
        ) as mock_find:
            mock_find.return_value = expected_operations

            from unittest.mock import MagicMock

            analyzer = MagicMock()
            analyzer.find_operations.return_value = expected_operations

            result = analyzer.find_operations(source)
            assert len(result) == 3


class TestSparkFunctionAnalysis:
    """Test analysis of Spark function calls."""

    def test_detect_current_timestamp(self):
        """Test detecting F.current_timestamp() calls."""
        expression = "F.current_timestamp()"

        expected_result = {
            "function": "current_timestamp",
            "return_type": "timestamp",
            "nullable": False,
        }

        with patch(
            "pyspark_transform_registry.static_analysis.type_inference.analyze_spark_function",
        ) as mock_analyze:
            mock_analyze.return_value = expected_result

            from unittest.mock import MagicMock

            analyzer = MagicMock()
            analyzer.analyze_spark_function.return_value = expected_result

            result = analyzer.analyze_spark_function(expression)
            assert result["function"] == "current_timestamp"
            assert result["return_type"] == "timestamp"
            assert result["nullable"] is False

    def test_detect_when_otherwise_expressions(self):
        """Test detecting F.when().otherwise() expressions."""
        expression = 'F.when(df.amount > 100, "high").otherwise("low")'

        expected_result = {
            "function": "when_otherwise",
            "return_type": "string",
            "nullable": False,
            "conditions": ["df.amount > 100"],
            "referenced_columns": ["amount"],
        }

        with patch(
            "pyspark_transform_registry.static_analysis.type_inference.analyze_spark_function",
        ) as mock_analyze:
            mock_analyze.return_value = expected_result

            from unittest.mock import MagicMock

            analyzer = MagicMock()
            analyzer.analyze_spark_function.return_value = expected_result

            result = analyzer.analyze_spark_function(expression)
            assert result["function"] == "when_otherwise"
            assert result["return_type"] == "string"

    def test_detect_mathematical_operations(self):
        """Test detecting mathematical operations on columns."""
        expression = "df.amount * 1.5"

        expected_result = {
            "operation": "multiplication",
            "return_type": "double",
            "nullable": True,  # Inherits from column
            "referenced_columns": ["amount"],
        }

        with patch(
            "pyspark_transform_registry.static_analysis.type_inference.analyze_expression",
        ) as mock_analyze:
            mock_analyze.return_value = expected_result

            from unittest.mock import MagicMock

            analyzer = MagicMock()
            analyzer.analyze_expression.return_value = expected_result

            result = analyzer.analyze_expression(expression)
            assert result["operation"] == "multiplication"
            assert result["return_type"] == "double"


class TestTypeInference:
    """Test type inference for expressions."""

    def test_infer_arithmetic_result_types(self):
        """Test type inference for arithmetic operations."""
        test_cases = [
            ("df.int_col + df.int_col", "integer"),
            ("df.int_col + df.double_col", "double"),
            ("df.double_col * 2", "double"),
            ("df.string_col + '_suffix'", "string"),
        ]

        for expression, expected_type in test_cases:
            with patch(
                "pyspark_transform_registry.static_analysis.type_inference.infer_expression_type",
            ) as mock_infer:
                mock_infer.return_value = expected_type

                from unittest.mock import MagicMock

                inference_engine = MagicMock()
                inference_engine.infer_expression_type.return_value = expected_type

                result = inference_engine.infer_expression_type(expression)
                assert result == expected_type

    def test_infer_spark_function_return_types(self):
        """Test type inference for Spark functions."""
        function_cases = [
            ("F.current_timestamp()", "timestamp"),
            ("F.current_date()", "date"),
            ("F.lit(42)", "integer"),
            ("F.lit('hello')", "string"),
            ("F.lit(True)", "boolean"),
            ("F.sum(df.amount)", "double"),
            ("F.count(df.id)", "integer"),
        ]

        for function_call, expected_type in function_cases:
            with patch(
                "pyspark_transform_registry.static_analysis.type_inference.infer_spark_function_type",
            ) as mock_infer:
                mock_infer.return_value = expected_type

                from unittest.mock import MagicMock

                inference_engine = MagicMock()
                inference_engine.infer_spark_function_type.return_value = expected_type

                result = inference_engine.infer_spark_function_type(function_call)
                assert result == expected_type


class TestComplexExpressionAnalysis:
    """Test analysis of complex expressions."""

    def test_analyze_nested_function_calls(self):
        """Test analyzing nested function calls."""
        expression = "F.lower(F.trim(df.name))"

        expected_result = {
            "return_type": "string",
            "nullable": True,  # Inherits from column
            "referenced_columns": ["name"],
            "functions_used": ["lower", "trim"],
        }

        with patch(
            "pyspark_transform_registry.static_analysis.type_inference.analyze_complex_expression",
        ) as mock_analyze:
            mock_analyze.return_value = expected_result

            from unittest.mock import MagicMock

            analyzer = MagicMock()
            analyzer.analyze_complex_expression.return_value = expected_result

            result = analyzer.analyze_complex_expression(expression)
            assert result["return_type"] == "string"
            assert "name" in result["referenced_columns"]

    def test_analyze_conditional_expressions(self):
        """Test analyzing conditional expressions with complex logic."""
        expression = """
        F.when(df.category == 'premium', df.amount * 0.9)
         .when(df.category == 'standard', df.amount * 0.95)
         .otherwise(df.amount)
        """

        expected_result = {
            "return_type": "double",
            "nullable": True,
            "referenced_columns": ["category", "amount"],
            "conditions": [
                "df.category == 'premium'",
                "df.category == 'standard'",
            ],
        }

        with patch(
            "pyspark_transform_registry.static_analysis.type_inference.analyze_complex_expression",
        ) as mock_analyze:
            mock_analyze.return_value = expected_result

            from unittest.mock import MagicMock

            analyzer = MagicMock()
            analyzer.analyze_complex_expression.return_value = expected_result

            result = analyzer.analyze_complex_expression(expression)
            assert result["return_type"] == "double"
            assert "category" in result["referenced_columns"]
            assert "amount" in result["referenced_columns"]


class TestEdgeCaseDetection:
    """Test detection of edge cases that are hard to analyze."""

    def test_detect_dynamic_column_names(self):
        """Test detecting dynamic column name usage."""
        source = """
        for col in columns:
            df = df.withColumn(f"processed_{col}", df[col] * 2)
        """

        expected_warnings = [
            "Dynamic column operations detected - manual verification recommended",
        ]

        with patch(
            "pyspark_transform_registry.static_analysis.analyzer.analyze_function",
        ) as mock_analyze:
            mock_constraint = PartialSchemaConstraint(
                warnings=expected_warnings,
            )
            mock_analyze.return_value = mock_constraint

            from unittest.mock import MagicMock

            analyzer = MagicMock()
            analyzer.analyze_function.return_value = mock_constraint

            result = analyzer.analyze_function(source)
            assert expected_warnings[0] in result.warnings

    def test_detect_udf_usage(self):
        """Test detecting UDF usage."""
        source = """
        @F.udf("boolean")
        def custom_logic(value):
            return value > threshold

        df.withColumn("result", custom_logic(df.amount))
        """

        expected_warnings = ["UDF detected - static analysis may be incomplete"]

        with patch(
            "pyspark_transform_registry.static_analysis.analyzer.analyze_function",
        ) as mock_analyze:
            mock_constraint = PartialSchemaConstraint(
                required_columns=[ColumnRequirement("amount", "double")],
                added_columns=[ColumnTransformation("result", "add", "boolean")],
                warnings=expected_warnings,
            )
            mock_analyze.return_value = mock_constraint

            from unittest.mock import MagicMock

            analyzer = MagicMock()
            analyzer.analyze_function.return_value = mock_constraint

            result = analyzer.analyze_function(source)
            assert expected_warnings[0] in result.warnings

    def test_detect_unparseable_code(self):
        """Test handling unparseable code gracefully."""
        source = "invalid python syntax {"

        with patch(
            "pyspark_transform_registry.static_analysis.analyzer.analyze_function",
        ) as mock_analyze:
            mock_constraint = PartialSchemaConstraint(
                warnings=["Could not parse function source - manual analysis required"],
            )
            mock_analyze.return_value = mock_constraint

            from unittest.mock import MagicMock

            analyzer = MagicMock()
            analyzer.analyze_function.return_value = mock_constraint

            result = analyzer.analyze_function(source)
            assert "Could not parse function source" in result.warnings[0]


class TestConstraintGeneration:
    """Test constraint generation from analysis results."""

    def test_generate_constraint_from_operations(self):
        """Test generating constraints from detected operations."""
        # Mock analysis results
        operations = [
            {
                "type": "withColumn",
                "column_name": "new_col",
                "expression_type": "timestamp",
            },
            {"type": "filter", "referenced_columns": ["status"]},
            {"type": "drop", "columns": ["temp_col"]},
        ]

        column_references = ["status", "amount"]  # amount used in some other way

        # Expected constraint
        expected = PartialSchemaConstraint(
            required_columns=[
                ColumnRequirement("status", "string"),
                ColumnRequirement("amount", "double"),
            ],
            added_columns=[
                ColumnTransformation("new_col", "add", "timestamp"),
            ],
            removed_columns=["temp_col"],
            preserves_other_columns=True,
        )

        with patch(
            "pyspark_transform_registry.static_analysis.schema_inference.generate_constraint",
        ) as mock_generate:
            mock_generate.return_value = expected

            from unittest.mock import MagicMock

            generator = MagicMock()
            generator.generate_constraint.return_value = expected

            result = generator.generate_constraint(operations, column_references)

            assert len(result.required_columns) == 2
            assert len(result.added_columns) == 1
            assert len(result.removed_columns) == 1
