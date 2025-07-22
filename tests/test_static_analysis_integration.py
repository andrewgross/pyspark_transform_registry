"""
Integration tests for the static analysis system.

This module tests the complete static analysis pipeline from function
analysis to constraint generation using real transform function examples.
"""

import pytest

from pyspark_transform_registry.schema_constraints import PartialSchemaConstraint
from pyspark_transform_registry.static_analysis import analyze_function
from tests.test_data.schema_constraint_examples import (
    EXPECTED_ADD_TIMESTAMP,
    EXPECTED_CALCULATE_METRICS,
    EXPECTED_CLEAN_TEXT_DATA,
    EXPECTED_CUSTOMER_ANALYTICS,
    EXPECTED_FILTER_ACTIVE,
    EXPECTED_NORMALIZE_AMOUNTS,
    add_timestamp,
    calculate_metrics,
    clean_text_data,
    customer_analytics,
    filter_active,
    normalize_amounts,
)


class TestStaticAnalysisIntegration:
    """Test the complete static analysis pipeline."""

    def test_analysis_produces_serializable_constraints(self):
        """Test that analysis produces constraints that can be serialized."""
        constraint = analyze_function(add_timestamp)

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
        constraint1 = analyze_function(add_timestamp)
        constraint2 = analyze_function(normalize_amounts)

        # Should be able to merge constraints
        merged = constraint1.merge_with(constraint2)

        assert isinstance(merged, PartialSchemaConstraint)
        assert merged.analysis_method == "merged"

    @pytest.mark.parametrize(
        "func,expected",
        [
            (add_timestamp, EXPECTED_ADD_TIMESTAMP),
            (normalize_amounts, EXPECTED_NORMALIZE_AMOUNTS),
            (filter_active, EXPECTED_FILTER_ACTIVE),
            (customer_analytics, EXPECTED_CUSTOMER_ANALYTICS),
            (clean_text_data, EXPECTED_CLEAN_TEXT_DATA),
            (calculate_metrics, EXPECTED_CALCULATE_METRICS),
        ],
    )
    def test_analysis_produces_reasonable_constraints(self, func, expected):
        """Test that analysis produces constraints that are reasonable compared to expected."""
        constraint = analyze_function(func)
        assert constraint == expected
