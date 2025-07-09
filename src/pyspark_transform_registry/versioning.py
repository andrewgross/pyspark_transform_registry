"""Semantic versioning utilities for transform functions."""

import re
from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class SemanticVersion:
    """Represents a semantic version with major.minor.patch format."""

    major: int
    minor: int
    patch: int

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SemanticVersion):
            return NotImplemented
        return (self.major, self.minor, self.patch) == (
            other.major,
            other.minor,
            other.patch,
        )

    def __lt__(self, other: "SemanticVersion") -> bool:
        return (self.major, self.minor, self.patch) < (
            other.major,
            other.minor,
            other.patch,
        )

    def __le__(self, other: "SemanticVersion") -> bool:
        return self < other or self == other

    def __gt__(self, other: "SemanticVersion") -> bool:
        return not self <= other

    def __ge__(self, other: "SemanticVersion") -> bool:
        return not self < other


def parse_semantic_version(version_str: str) -> SemanticVersion:
    """Parse a semantic version string into components.

    Args:
        version_str: Version string in format "major.minor.patch"

    Returns:
        SemanticVersion object

    Raises:
        ValueError: If version string is invalid
    """
    pattern = r"^(\d+)\.(\d+)\.(\d+)$"
    match = re.match(pattern, version_str)

    if not match:
        raise ValueError(
            f"Invalid semantic version format: {version_str}. Expected format: major.minor.patch",
        )

    major, minor, patch = map(int, match.groups())
    return SemanticVersion(major, minor, patch)


def validate_semantic_version(version_str: str) -> bool:
    """Validate that a string follows semantic versioning format.

    Args:
        version_str: Version string to validate

    Returns:
        True if valid, False otherwise
    """
    try:
        parse_semantic_version(version_str)
        return True
    except ValueError:
        return False


def get_latest_version(transform_name: str) -> Optional[SemanticVersion]:
    """Get the latest semantic version for a transform function.

    Args:
        transform_name: Name of the transform function

    Returns:
        Latest semantic version or None if no versions exist
    """
    from .core import find_transform_versions

    # Use the model registry to find versions
    versions = find_transform_versions(name=transform_name, latest_only=True)

    if not versions:
        return None

    # Return the latest semantic version
    return versions[0]["semantic_version"]


def generate_next_version(
    transform_name: str,
    current_func: Callable,
    version_bump: Optional[str] = None,
) -> SemanticVersion:
    """Generate the next semantic version for a transform function.

    Args:
        transform_name: Name of the transform function
        current_func: Current function to analyze
        version_bump: Optional explicit version bump type ("major", "minor", "patch")

    Returns:
        Next semantic version
    """
    latest_version = get_latest_version(transform_name)

    # If no previous versions, start with 1.0.0
    if latest_version is None:
        return SemanticVersion(1, 0, 0)

    # If explicit version bump specified
    if version_bump:
        if version_bump == "major":
            return SemanticVersion(latest_version.major + 1, 0, 0)
        elif version_bump == "minor":
            return SemanticVersion(latest_version.major, latest_version.minor + 1, 0)
        elif version_bump == "patch":
            return SemanticVersion(
                latest_version.major,
                latest_version.minor,
                latest_version.patch + 1,
            )
        else:
            raise ValueError(f"Invalid version bump type: {version_bump}")

    # Auto-determine version bump based on interface changes
    interface_change = _analyze_interface_changes(transform_name, current_func)

    if interface_change == "major":
        return SemanticVersion(latest_version.major + 1, 0, 0)
    elif interface_change == "minor":
        return SemanticVersion(latest_version.major, latest_version.minor + 1, 0)
    else:
        # Default to minor version bump for new functionality
        return SemanticVersion(latest_version.major, latest_version.minor + 1, 0)


def _analyze_interface_changes(transform_name: str, current_func: Callable) -> str:
    """Analyze interface changes to determine version bump type.

    Args:
        transform_name: Name of the transform function
        current_func: Current function to analyze

    Returns:
        Version bump type: "major", "minor", or "patch"
    """
    from .metadata import _get_function_metadata

    # Get metadata for current function
    param_info, return_type, doc = _get_function_metadata(current_func)
    current_metadata = {
        "param_info": param_info,
        "return_type": return_type,
        "doc": doc,
    }

    # Get metadata for latest version
    latest_metadata = _get_latest_function_metadata(transform_name)

    if latest_metadata is None:
        # No previous version, this is a new function
        return "minor"

    # Check for breaking changes
    if _has_breaking_changes(latest_metadata, current_metadata):
        return "major"

    # Check for new functionality (compatible changes)
    if _has_compatible_changes(latest_metadata, current_metadata):
        return "minor"

    # No significant changes detected
    return "minor"  # Default to minor for new deployment


def _get_latest_function_metadata(transform_name: str) -> Optional[dict]:
    """Get metadata for the latest version of a transform function.

    Args:
        transform_name: Name of the transform function

    Returns:
        Function metadata dictionary or None if not found
    """
    from .core import find_transform_versions

    # Use the model registry to find the latest version
    versions = find_transform_versions(name=transform_name, latest_only=True)

    if not versions:
        return None

    latest_version = versions[0]

    # Parse param_info from JSON if it exists
    param_info = []
    if "param_info" in latest_version["metadata"]:
        import json

        try:
            param_info = json.loads(latest_version["metadata"]["param_info"])
        except (json.JSONDecodeError, TypeError):
            param_info = []

    return {
        "run_id": latest_version["run_id"],
        "param_info": param_info,
        "return_type": latest_version["metadata"].get("return_type"),
        "semantic_version": str(latest_version["semantic_version"]),
    }


def _extract_param_info_from_tags(tags: dict) -> list[dict]:
    """Extract parameter information from MLflow tags.

    Args:
        tags: Dictionary of MLflow tags

    Returns:
        List of parameter dictionaries
    """
    # This is a simplified implementation
    # In practice, we'd need to store parameter info in a structured way
    param_info = []

    # Look for parameter-related tags
    for key, value in tags.items():
        if key.startswith("param_"):
            param_name = key.replace("param_", "")
            param_info.append(
                {
                    "name": param_name,
                    "annotation": value,
                    "default": None,  # Would need to be stored separately
                },
            )

    return param_info


def _has_breaking_changes(old_metadata: dict, new_metadata: dict) -> bool:
    """Check if there are breaking changes between function versions.

    Args:
        old_metadata: Metadata from previous version
        new_metadata: Metadata from current version

    Returns:
        True if breaking changes detected
    """
    # Check return type changes
    if old_metadata.get("return_type") != new_metadata.get("return_type"):
        return True

    # Check for removed or changed required parameters
    old_params = {p["name"]: p for p in old_metadata.get("param_info", [])}
    new_params = {p["name"]: p for p in new_metadata.get("param_info", [])}

    # Check for removed parameters
    for param_name, param_info in old_params.items():
        if param_name not in new_params:
            # Parameter was removed
            if param_info.get("default") is None:
                # Required parameter was removed - breaking change
                return True
        else:
            # Parameter exists in both versions
            new_param = new_params[param_name]
            if param_info.get("annotation") != new_param.get("annotation"):
                # Parameter type changed - breaking change
                return True

    # Check for new required parameters
    for param_name, param_info in new_params.items():
        if param_name not in old_params:
            # New parameter was added
            if param_info.get("default") is None:
                # New required parameter - breaking change
                return True

    return False


def _has_compatible_changes(old_metadata: dict, new_metadata: dict) -> bool:
    """Check if there are compatible changes between function versions.

    Args:
        old_metadata: Metadata from previous version
        new_metadata: Metadata from current version

    Returns:
        True if compatible changes detected
    """
    old_params = {p["name"]: p for p in old_metadata.get("param_info", [])}
    new_params = {p["name"]: p for p in new_metadata.get("param_info", [])}

    # Check for new optional parameters
    for param_name, param_info in new_params.items():
        if param_name not in old_params:
            # New parameter added
            if param_info.get("default") is not None:
                # New optional parameter - compatible change
                return True

    # If no new parameters, assume there are functional improvements
    return True


def satisfies_version_constraint(version: SemanticVersion, constraint: str) -> bool:
    """Check if a version satisfies a version constraint.

    Args:
        version: Semantic version to check
        constraint: Version constraint (e.g., ">=1.0.0,<2.0.0")

    Returns:
        True if version satisfies constraint
    """
    # Parse constraint (simplified implementation)
    constraints = [c.strip() for c in constraint.split(",")]

    for constraint_part in constraints:
        if not _check_single_constraint(version, constraint_part):
            return False

    return True


def _check_single_constraint(version: SemanticVersion, constraint: str) -> bool:
    """Check if version satisfies a single constraint.

    Args:
        version: Semantic version to check
        constraint: Single constraint (e.g., ">=1.0.0")

    Returns:
        True if version satisfies constraint
    """
    # Parse operator and version
    operators = [">=", "<=", "==", "!=", ">", "<"]

    for op in operators:
        if constraint.startswith(op):
            constraint_version_str = constraint[len(op) :].strip()
            constraint_version = parse_semantic_version(constraint_version_str)

            if op == ">=":
                return version >= constraint_version
            elif op == "<=":
                return version <= constraint_version
            elif op == "==":
                return version == constraint_version
            elif op == "!=":
                return version != constraint_version
            elif op == ">":
                return version > constraint_version
            elif op == "<":
                return version < constraint_version

    # If no operator found, assume exact match
    constraint_version = parse_semantic_version(constraint)
    return version == constraint_version
