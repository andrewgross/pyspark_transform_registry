"""
Integration tests for the static analysis system.

This module tests the complete static analysis pipeline from function
analysis to constraint generation using real transform function examples.
"""

import pytest

from pyspark_transform_registry.schema_constraints import PartialSchemaConstraint
from pyspark_transform_registry.static_analysis import analyze_function
from tests.fixtures.schema_constraint_examples import (
    ALL_TRANSFORM_EXAMPLES,
    add_timestamp_f,
    normalize_amounts_f,
)


class TestStaticAnalysisIntegration:
    """Test the complete static analysis pipeline."""

    def test_analysis_produces_serializable_constraints(self):
        """Test that analysis produces constraints that can be serialized."""
        constraint = analyze_function(add_timestamp_f)

        # Should be able to serialize to JSON
        json_str = constraint.to_json()
        assert isinstance(json_str, str)
        assert len(json_str) > 0

        # Should be able to deserialize
        reconstructed = PartialSchemaConstraint.from_json(json_str)
        assert isinstance(reconstructed, PartialSchemaConstraint)
        assert reconstructed.analysis_method == constraint.analysis_method

    def test_constraint_merging_works(self):
        """Test that constraints from different functions can be merged."""
        constraint1 = analyze_function(add_timestamp_f)
        constraint2 = analyze_function(normalize_amounts_f)

        # Should be able to merge constraints
        merged = constraint1.merge_with(constraint2)

        assert isinstance(merged, PartialSchemaConstraint)
        assert merged.analysis_method == "merged"

    @pytest.mark.parametrize(
        "func,expected",
        ALL_TRANSFORM_EXAMPLES,
    )
    def test_analysis_produces_reasonable_constraints(self, func, expected):
        """Test that analysis produces constraints that are reasonable compared to expected."""
        constraint = analyze_function(func)
        assert constraint.to_dict() == expected.to_dict()
