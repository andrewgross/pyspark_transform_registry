"""
Schema constraint system for partial DataFrame validation.

This module provides classes and utilities for defining and working with
partial schema constraints that express requirements and transformations
without needing complete schema definitions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, order=True)
class ColumnRequirement:
    """
    Represents a requirement for a specific column in a DataFrame.

    This is used to specify that a transform function requires certain
    columns to be present with specific types and nullability constraints.
    """

    name: str
    type: str  # PySpark type names: "string", "double", "integer", "timestamp", etc.
    nullable: bool = True
    description: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "name": self.name,
            "type": self.type,
            "nullable": self.nullable,
        }
        if self.description:
            result["description"] = self.description
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ColumnRequirement:
        """Create from dictionary."""
        return cls(
            name=data["name"],
            type=data["type"],
            nullable=data.get("nullable", True),
            description=data.get("description"),
        )


@dataclass(frozen=True, order=True)
class ColumnTransformation:
    """
    Represents a transformation applied to a column by a transform function.

    This captures operations like adding new columns, modifying existing ones,
    or removing columns from the DataFrame.
    """

    name: str
    operation: str  # "add", "modify", "remove"
    type: str | None = None  # Required for "add" and "modify" operations
    nullable: bool = True
    description: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "name": self.name,
            "operation": self.operation,
            "nullable": self.nullable,
        }
        if self.type:
            result["type"] = self.type
        if self.description:
            result["description"] = self.description
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ColumnTransformation:
        """Create from dictionary."""
        return cls(
            name=data["name"],
            operation=data["operation"],
            type=data.get("type"),
            nullable=data.get("nullable", True),
            description=data.get("description"),
        )


@dataclass
class PartialSchemaConstraint:
    """
    Represents partial schema constraints for a PySpark transform function.

    Unlike complete schema definitions, this captures only the constraints
    that a transform function actually requires and the transformations it
    performs. This allows for flexible validation without over-specifying
    input requirements.
    """

    schema_version: str = "1.0"
    required_columns: list[ColumnRequirement] = field(default_factory=list)
    optional_columns: list[ColumnRequirement] = field(default_factory=list)
    added_columns: list[ColumnTransformation] = field(default_factory=list)
    modified_columns: list[ColumnTransformation] = field(default_factory=list)
    removed_columns: list[str] = field(default_factory=list)
    preserves_other_columns: bool = True
    analysis_method: str = "static_analysis"  # "static_analysis", "manual", "hybrid"
    warnings: list[str] = field(default_factory=list)

    def __eq__(self, other):
        if not isinstance(other, PartialSchemaConstraint):
            return False

        return (
            set(self.required_columns) == set(other.required_columns)
            and set(self.optional_columns) == set(other.optional_columns)
            and set(self.added_columns) == set(other.added_columns)
            and set(self.modified_columns) == set(other.modified_columns)
            and set(self.removed_columns) == set(other.removed_columns)
            and self.preserves_other_columns == other.preserves_other_columns
            and self.analysis_method == other.analysis_method
            and set(self.warnings) == set(other.warnings)
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "schema_version": self.schema_version,
            "required_columns": [
                col.to_dict() for col in sorted(self.required_columns)
            ],
            "optional_columns": [
                col.to_dict() for col in sorted(self.optional_columns)
            ],
            "added_columns": [col.to_dict() for col in sorted(self.added_columns)],
            "modified_columns": [
                col.to_dict() for col in sorted(self.modified_columns)
            ],
            "removed_columns": sorted(self.removed_columns),
            "preserves_other_columns": self.preserves_other_columns,
            "analysis_method": self.analysis_method,
            "warnings": sorted(self.warnings),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PartialSchemaConstraint:
        """Create from dictionary."""
        return cls(
            schema_version=data.get("schema_version", "1.0"),
            required_columns=[
                ColumnRequirement.from_dict(col)
                for col in data.get("required_columns", [])
            ],
            optional_columns=[
                ColumnRequirement.from_dict(col)
                for col in data.get("optional_columns", [])
            ],
            added_columns=[
                ColumnTransformation.from_dict(col)
                for col in data.get("added_columns", [])
            ],
            modified_columns=[
                ColumnTransformation.from_dict(col)
                for col in data.get("modified_columns", [])
            ],
            removed_columns=data.get("removed_columns", []),
            preserves_other_columns=data.get("preserves_other_columns", True),
            analysis_method=data.get("analysis_method", "static_analysis"),
            warnings=data.get("warnings", []),
        )

    def to_json(self) -> str:
        """Convert to JSON string for storage."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> PartialSchemaConstraint:
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))

    def add_warning(self, message: str) -> None:
        """Add a warning message to the constraints."""
        if message not in self.warnings:
            self.warnings.append(message)

    def merge_with(self, other: PartialSchemaConstraint) -> PartialSchemaConstraint:
        """
        Merge with another constraint, typically used for function composition.

        Args:
            other: Another constraint to merge with

        Returns:
            A new merged constraint
        """
        # Merge required columns (union of requirements)
        required_cols = {col.name: col for col in self.required_columns}
        for col in other.required_columns:
            if col.name in required_cols:
                # If both require the same column, use the more restrictive constraint
                existing = required_cols[col.name]
                if not col.nullable and existing.nullable:
                    required_cols[col.name] = col
            else:
                required_cols[col.name] = col

        # Merge transformations (output of first becomes input to second)
        added_cols = list(self.added_columns)
        modified_cols = {col.name: col for col in self.modified_columns}

        # Apply second constraint's transformations
        for col in other.added_columns:
            added_cols.append(col)

        for col in other.modified_columns:
            modified_cols[col.name] = col

        # Combine removed columns
        removed_cols = list(set(self.removed_columns + other.removed_columns))

        # Merge warnings
        warnings = list(set(self.warnings + other.warnings))

        return PartialSchemaConstraint(
            schema_version=self.schema_version,
            required_columns=list(required_cols.values()),
            optional_columns=self.optional_columns + other.optional_columns,
            added_columns=added_cols,
            modified_columns=list(modified_cols.values()),
            removed_columns=removed_cols,
            preserves_other_columns=self.preserves_other_columns
            and other.preserves_other_columns,
            analysis_method="merged",
            warnings=warnings,
        )

    def get_all_required_columns(self) -> list[str]:
        """Get names of all required columns."""
        return [col.name for col in self.required_columns]

    def get_all_output_columns(self) -> list[str]:
        """
        Get names of all columns that will be in the output.
        Note: This doesn't include preserved columns if preserves_other_columns=True.
        """
        output_cols = []

        # Add required columns (they pass through)
        output_cols.extend(self.get_all_required_columns())

        # Add optional columns (they pass through if present)
        output_cols.extend([col.name for col in self.optional_columns])

        # Add newly added columns
        output_cols.extend([col.name for col in self.added_columns])

        # Modified columns are already in the required/optional lists

        # Remove deleted columns
        output_cols = [col for col in output_cols if col not in self.removed_columns]

        return list(set(output_cols))  # Remove duplicates


# Validation result classes
@dataclass
class ValidationIssue:
    """Represents an issue found during schema validation."""

    severity: str  # "error", "warning", "info"
    message: str
    column_name: str | None = None
    expected_type: str | None = None
    actual_type: str | None = None


@dataclass
class ValidationResult:
    """Result of schema constraint validation."""

    is_valid: bool
    issues: list[ValidationIssue] = field(default_factory=list)

    def add_error(
        self,
        message: str,
        column_name: str | None = None,
        expected_type: str | None = None,
        actual_type: str | None = None,
    ) -> None:
        """Add an error to the validation result."""
        self.issues.append(
            ValidationIssue(
                severity="error",
                message=message,
                column_name=column_name,
                expected_type=expected_type,
                actual_type=actual_type,
            ),
        )
        self.is_valid = False

    def add_warning(self, message: str, column_name: str | None = None) -> None:
        """Add a warning to the validation result."""
        self.issues.append(
            ValidationIssue(
                severity="warning",
                message=message,
                column_name=column_name,
            ),
        )

    def get_error_messages(self) -> list[str]:
        """Get all error messages."""
        return [issue.message for issue in self.issues if issue.severity == "error"]

    def get_warning_messages(self) -> list[str]:
        """Get all warning messages."""
        return [issue.message for issue in self.issues if issue.severity == "warning"]


# Utility functions for working with PySpark types
PYSPARK_TYPE_MAPPING = {
    # Python types to PySpark type strings
    str: "string",
    int: "integer",
    float: "double",
    bool: "boolean",
    bytes: "binary",
    # Common type aliases
    "str": "string",
    "int": "integer",
    "float": "double",
    "bool": "boolean",
    "bytes": "binary",
}


def python_type_to_pyspark_type(python_type: type | str) -> str:
    """
    Convert Python type to PySpark type string.

    Args:
        python_type: Python type or type name

    Returns:
        PySpark type string
    """
    if isinstance(python_type, str):
        return PYSPARK_TYPE_MAPPING.get(python_type, python_type)
    return PYSPARK_TYPE_MAPPING.get(python_type, str(python_type))


def infer_pyspark_type_from_value(value: Any) -> str:
    """
    Infer PySpark type from a Python value.

    Args:
        value: Python value

    Returns:
        PySpark type string
    """
    if value is None:
        return "string"  # Default to string for None
    return python_type_to_pyspark_type(type(value))
