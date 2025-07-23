"""
Tests for the schema constraint system.

This module tests the constraint data structures, serialization,
validation logic, and constraint merging functionality.
"""

import json

from pyspark_transform_registry.schema_constraints import (
    ColumnRequirement,
    ColumnTransformation,
    PartialSchemaConstraint,
    ValidationResult,
    infer_pyspark_type_from_value,
    python_type_to_pyspark_type,
)


class TestColumnRequirement:
    """Test ColumnRequirement data structure."""

    def test_column_requirement_creation(self):
        """Test basic ColumnRequirement creation."""
        col_req = ColumnRequirement(
            "customer_id",
            "string",
            nullable=False,
            description="Primary key",
        )

        assert col_req.name == "customer_id"
        assert col_req.type == "string"
        assert col_req.nullable is False
        assert col_req.description == "Primary key"

    def test_column_requirement_defaults(self):
        """Test ColumnRequirement defaults."""
        col_req = ColumnRequirement("amount", "double")

        assert col_req.nullable is True
        assert col_req.description is None

    def test_column_requirement_serialization(self):
        """Test ColumnRequirement to_dict/from_dict."""
        original = ColumnRequirement(
            "order_count",
            "integer",
            nullable=False,
            description="Number of orders",
        )

        # Test to_dict
        data = original.to_dict()
        expected = {
            "name": "order_count",
            "type": "integer",
            "nullable": False,
            "description": "Number of orders",
        }
        assert data == expected

        # Test from_dict
        reconstructed = ColumnRequirement.from_dict(data)
        assert reconstructed.name == original.name
        assert reconstructed.type == original.type
        assert reconstructed.nullable == original.nullable
        assert reconstructed.description == original.description

    def test_column_requirement_serialization_minimal(self):
        """Test ColumnRequirement serialization with minimal data."""
        original = ColumnRequirement("status", "string")

        data = original.to_dict()
        expected = {
            "name": "status",
            "type": "string",
            "nullable": True,
        }
        assert data == expected

        reconstructed = ColumnRequirement.from_dict(data)
        assert reconstructed.description is None


class TestColumnTransformation:
    """Test ColumnTransformation data structure."""

    def test_column_transformation_creation(self):
        """Test basic ColumnTransformation creation."""
        transform = ColumnTransformation(
            "new_column",
            "add",
            "timestamp",
            nullable=False,
            description="Creation time",
        )

        assert transform.name == "new_column"
        assert transform.operation == "add"
        assert transform.type == "timestamp"
        assert transform.nullable is False
        assert transform.description == "Creation time"

    def test_column_transformation_remove_operation(self):
        """Test remove operation doesn't need type."""
        transform = ColumnTransformation("temp_col", "remove")

        assert transform.name == "temp_col"
        assert transform.operation == "remove"
        assert transform.type is None
        assert transform.nullable is True

    def test_column_transformation_serialization(self):
        """Test ColumnTransformation to_dict/from_dict."""
        original = ColumnTransformation(
            "processed_amount",
            "modify",
            "double",
            nullable=False,
            description="Processed value",
        )

        data = original.to_dict()
        expected = {
            "name": "processed_amount",
            "operation": "modify",
            "type": "double",
            "nullable": False,
            "description": "Processed value",
        }
        assert data == expected

        reconstructed = ColumnTransformation.from_dict(data)
        assert reconstructed.name == original.name
        assert reconstructed.operation == original.operation
        assert reconstructed.type == original.type
        assert reconstructed.nullable == original.nullable
        assert reconstructed.description == original.description


class TestPartialSchemaConstraint:
    """Test PartialSchemaConstraint data structure and operations."""

    def test_constraint_creation_defaults(self):
        """Test PartialSchemaConstraint creation with defaults."""
        constraint = PartialSchemaConstraint()

        assert constraint.schema_version == "1.0"
        assert constraint.required_columns == []
        assert constraint.optional_columns == []
        assert constraint.added_columns == []
        assert constraint.modified_columns == []
        assert constraint.removed_columns == []
        assert constraint.preserves_other_columns is True
        assert constraint.analysis_method == "static_analysis"
        assert constraint.warnings == []

    def test_constraint_creation_with_data(self):
        """Test PartialSchemaConstraint with real data."""
        constraint = PartialSchemaConstraint(
            required_columns=[
                ColumnRequirement("customer_id", "string", nullable=False),
                ColumnRequirement("amount", "double"),
            ],
            added_columns=[
                ColumnTransformation(
                    "processed_at",
                    "add",
                    "timestamp",
                    nullable=False,
                ),
            ],
        )

        assert len(constraint.required_columns) == 2
        assert len(constraint.added_columns) == 1

    def test_constraint_json_serialization(self):
        """Test PartialSchemaConstraint JSON serialization."""
        original = PartialSchemaConstraint(
            required_columns=[ColumnRequirement("id", "integer")],
            added_columns=[ColumnTransformation("flag", "add", "boolean")],
            removed_columns=["temp"],
            warnings=["Static analysis limitation"],
        )

        # Test to_json
        json_str = original.to_json()
        json_data = json.loads(json_str)

        assert json_data["schema_version"] == "1.0"
        assert len(json_data["required_columns"]) == 1
        assert len(json_data["added_columns"]) == 1
        assert json_data["removed_columns"] == ["temp"]
        assert json_data["warnings"] == ["Static analysis limitation"]

        # Test from_json
        reconstructed = PartialSchemaConstraint.from_json(json_str)
        assert reconstructed.schema_version == original.schema_version
        assert len(reconstructed.required_columns) == len(original.required_columns)
        assert (
            reconstructed.required_columns[0].name == original.required_columns[0].name
        )
        assert len(reconstructed.added_columns) == len(original.added_columns)
        assert reconstructed.removed_columns == original.removed_columns
        assert reconstructed.warnings == original.warnings

    def test_add_warning(self):
        """Test adding warnings to constraints."""
        constraint = PartialSchemaConstraint()

        constraint.add_warning("First warning")
        assert constraint.warnings == ["First warning"]

        constraint.add_warning("Second warning")
        assert constraint.warnings == ["First warning", "Second warning"]

        # Adding same warning shouldn't duplicate
        constraint.add_warning("First warning")
        assert constraint.warnings == ["First warning", "Second warning"]

    def test_get_all_required_columns(self):
        """Test getting all required column names."""
        constraint = PartialSchemaConstraint(
            required_columns=[
                ColumnRequirement("col1", "string"),
                ColumnRequirement("col2", "integer"),
                ColumnRequirement("col3", "double"),
            ],
        )

        required = constraint.get_all_required_columns()
        assert required == ["col1", "col2", "col3"]

    def test_get_all_output_columns(self):
        """Test getting all output column names."""
        constraint = PartialSchemaConstraint(
            required_columns=[
                ColumnRequirement("existing1", "string"),
                ColumnRequirement("existing2", "integer"),
            ],
            optional_columns=[
                ColumnRequirement("optional1", "string"),
            ],
            added_columns=[
                ColumnTransformation("new1", "add", "timestamp"),
                ColumnTransformation("new2", "add", "boolean"),
            ],
            removed_columns=["existing2"],  # Remove one of the required columns
        )

        output_cols = constraint.get_all_output_columns()
        expected = ["existing1", "optional1", "new1", "new2"]  # existing2 removed
        assert set(output_cols) == set(expected)

    def test_constraint_merging(self):
        """Test merging two constraints."""
        constraint1 = PartialSchemaConstraint(
            required_columns=[
                ColumnRequirement("customer_id", "string", nullable=False),
                ColumnRequirement("amount", "double"),
            ],
            added_columns=[
                ColumnTransformation("normalized_amount", "add", "double"),
            ],
        )

        constraint2 = PartialSchemaConstraint(
            required_columns=[
                ColumnRequirement(
                    "amount",
                    "double",
                    nullable=False,
                ),  # More restrictive
                ColumnRequirement("category", "string"),
            ],
            added_columns=[
                ColumnTransformation("category_flag", "add", "boolean"),
            ],
            modified_columns=[
                ColumnTransformation("normalized_amount", "modify", "double"),
            ],
            removed_columns=["temp_col"],
            warnings=["Complex logic detected"],
        )

        merged = constraint1.merge_with(constraint2)

        # Check required columns merge (more restrictive wins)
        required_names = {col.name: col for col in merged.required_columns}
        assert "customer_id" in required_names
        assert "amount" in required_names
        assert "category" in required_names
        assert required_names["amount"].nullable is False  # More restrictive

        # Check added columns
        added_names = [col.name for col in merged.added_columns]
        assert "normalized_amount" in added_names
        assert "category_flag" in added_names

        # Check modified columns
        modified_names = [col.name for col in merged.modified_columns]
        assert "normalized_amount" in modified_names

        # Check other properties
        assert merged.removed_columns == ["temp_col"]
        assert merged.analysis_method == "merged"
        assert "Complex logic detected" in merged.warnings


class TestValidationResult:
    """Test ValidationResult data structure."""

    def test_validation_result_creation(self):
        """Test ValidationResult creation."""
        result = ValidationResult(is_valid=True)

        assert result.is_valid is True
        assert result.issues == []

    def test_add_error(self):
        """Test adding errors to validation result."""
        result = ValidationResult(is_valid=True)

        result.add_error(
            "Missing column 'amount'",
            column_name="amount",
            expected_type="double",
        )

        assert result.is_valid is False
        assert len(result.issues) == 1

        issue = result.issues[0]
        assert issue.severity == "error"
        assert issue.message == "Missing column 'amount'"
        assert issue.column_name == "amount"
        assert issue.expected_type == "double"

    def test_add_warning(self):
        """Test adding warnings to validation result."""
        result = ValidationResult(is_valid=True)

        result.add_warning("Column 'extra' will be ignored", column_name="extra")

        assert result.is_valid is True  # Warnings don't make result invalid
        assert len(result.issues) == 1

        issue = result.issues[0]
        assert issue.severity == "warning"
        assert issue.message == "Column 'extra' will be ignored"
        assert issue.column_name == "extra"

    def test_get_error_and_warning_messages(self):
        """Test getting error and warning messages."""
        result = ValidationResult(is_valid=True)

        result.add_error("Error 1")
        result.add_error("Error 2")
        result.add_warning("Warning 1")
        result.add_warning("Warning 2")

        errors = result.get_error_messages()
        warnings = result.get_warning_messages()

        assert errors == ["Error 1", "Error 2"]
        assert warnings == ["Warning 1", "Warning 2"]


class TestUtilityFunctions:
    """Test utility functions for type handling."""

    def test_python_type_to_pyspark_type(self):
        """Test Python type to PySpark type conversion."""
        # Test type objects
        assert python_type_to_pyspark_type(str) == "string"
        assert python_type_to_pyspark_type(int) == "integer"
        assert python_type_to_pyspark_type(float) == "double"
        assert python_type_to_pyspark_type(bool) == "boolean"
        assert python_type_to_pyspark_type(bytes) == "binary"

        # Test string type names
        assert python_type_to_pyspark_type("str") == "string"
        assert python_type_to_pyspark_type("int") == "integer"
        assert python_type_to_pyspark_type("float") == "double"
        assert python_type_to_pyspark_type("bool") == "boolean"

        # Test unknown types
        assert python_type_to_pyspark_type("unknown") == "unknown"

        class CustomType:
            pass

        result = python_type_to_pyspark_type(CustomType)
        assert "CustomType" in result

    def test_infer_pyspark_type_from_value(self):
        """Test PySpark type inference from values."""
        assert infer_pyspark_type_from_value("hello") == "string"
        assert infer_pyspark_type_from_value(42) == "integer"
        assert infer_pyspark_type_from_value(3.14) == "double"
        assert infer_pyspark_type_from_value(True) == "boolean"
        assert infer_pyspark_type_from_value(b"bytes") == "binary"
        assert infer_pyspark_type_from_value(None) == "string"  # Default for None


class TestConstraintComplexScenarios:
    """Test complex constraint scenarios."""

    def test_complex_constraint_serialization(self):
        """Test serialization of complex constraint with all features."""
        constraint = PartialSchemaConstraint(
            schema_version="1.0",
            required_columns=[
                ColumnRequirement(
                    "id",
                    "integer",
                    nullable=False,
                    description="Primary key",
                ),
                ColumnRequirement(
                    "amount",
                    "double",
                    nullable=True,
                    description="Transaction amount",
                ),
            ],
            optional_columns=[
                ColumnRequirement(
                    "category",
                    "string",
                    nullable=True,
                    description="Category code",
                ),
            ],
            added_columns=[
                ColumnTransformation(
                    "processed_at",
                    "add",
                    "timestamp",
                    nullable=False,
                    description="Processing timestamp",
                ),
                ColumnTransformation(
                    "is_valid",
                    "add",
                    "boolean",
                    nullable=False,
                    description="Validation flag",
                ),
            ],
            modified_columns=[
                ColumnTransformation(
                    "amount",
                    "modify",
                    "double",
                    nullable=False,
                    description="Normalized amount",
                ),
            ],
            removed_columns=["temp_staging", "debug_info"],
            preserves_other_columns=True,
            analysis_method="hybrid",
            warnings=["UDF detected", "Complex conditional logic"],
        )

        # Test full serialization round trip
        json_str = constraint.to_json()
        reconstructed = PartialSchemaConstraint.from_json(json_str)

        # Verify all properties
        assert reconstructed.schema_version == constraint.schema_version
        assert len(reconstructed.required_columns) == len(constraint.required_columns)
        assert len(reconstructed.optional_columns) == len(constraint.optional_columns)
        assert len(reconstructed.added_columns) == len(constraint.added_columns)
        assert len(reconstructed.modified_columns) == len(constraint.modified_columns)
        assert set(reconstructed.removed_columns) == set(constraint.removed_columns)
        assert (
            reconstructed.preserves_other_columns == constraint.preserves_other_columns
        )
        assert reconstructed.analysis_method == constraint.analysis_method
        assert set(reconstructed.warnings) == set(constraint.warnings)

        # Verify nested object details
        assert any(
            col.description == "Primary key" for col in reconstructed.required_columns
        )
        assert any(
            col.description == "Processing timestamp"
            for col in reconstructed.added_columns
        )

    def test_constraint_with_duplicate_column_names(self):
        """Test handling of constraints with duplicate column names."""
        # This tests edge cases in merging and validation
        constraint1 = PartialSchemaConstraint(
            required_columns=[ColumnRequirement("amount", "double")],
            added_columns=[
                ColumnTransformation("amount", "add", "double"),
            ],  # Conflict!
        )

        constraint2 = PartialSchemaConstraint(
            modified_columns=[ColumnTransformation("amount", "modify", "double")],
        )

        # Merging should handle conflicts gracefully
        merged = constraint1.merge_with(constraint2)

        # The modified operation should win over add when merging
        modified_names = [col.name for col in merged.modified_columns]
        assert "amount" in modified_names

    def test_empty_constraint_operations(self):
        """Test operations on empty constraints."""
        empty = PartialSchemaConstraint()

        assert empty.get_all_required_columns() == []
        assert empty.get_all_output_columns() == []

        # Merging with empty should preserve original
        constraint = PartialSchemaConstraint(
            required_columns=[ColumnRequirement("test", "string")],
        )

        merged = constraint.merge_with(empty)
        assert len(merged.required_columns) == 1
        assert merged.required_columns[0].name == "test"
