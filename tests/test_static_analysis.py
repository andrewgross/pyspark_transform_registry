"""
Tests for the static analysis engine.

This module tests the AST-based static analysis system that infers
schema constraints from PySpark transform function source code.
"""

from pyspark_transform_registry.schema_constraints import (
    ColumnRequirement,
    ColumnTransformation,
)
from pyspark_transform_registry.static_analysis import analyze_function
from tests.fixtures.schema_constraint_examples import (
    add_timestamp_f,
    normalize_amounts_f,
)


class TestFullFunctionAnalysis:
    """Test full function analysis."""

    def test_analyze_function(self):
        """Test analyzing a full function."""

        result = analyze_function(normalize_amounts_f)
        assert result.required_columns == [ColumnRequirement("amount", "double")]
        assert result.added_columns == []
        assert result.modified_columns == [
            ColumnTransformation("amount", "modify", "double"),
        ]
        assert result.removed_columns == []
        assert result.preserves_other_columns
        assert result.warnings == []

    def test_analyze_timestamp_function(self):
        """Test analyzing a full function with timestamp."""

        result = analyze_function(add_timestamp_f)
        assert result.required_columns == []  # No columns are required
        assert result.added_columns == [
            ColumnTransformation(
                "created_at",
                "add",
                "timestamp",
                False,
            ),  # Should add created_at with timestamp type, not nullable
        ]
        assert result.modified_columns == []
        assert result.removed_columns == []
        assert result.preserves_other_columns
        assert result.warnings == []
        assert result.removed_columns == []
        assert result.preserves_other_columns
        assert result.warnings == []
        assert result.removed_columns == []
        assert result.preserves_other_columns
        assert result.warnings == []
