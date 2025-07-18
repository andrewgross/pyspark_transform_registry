"""Tests for semantic versioning functionality."""

import pytest
import mlflow
from pyspark.sql import DataFrame

from pyspark_transform_registry import (
    SemanticVersion,
    parse_semantic_version,
    validate_semantic_version,
    get_latest_version,
    satisfies_version_constraint,
    log_transform_function,
    find_transform_versions,
)


class TestSemanticVersion:
    """Test SemanticVersion class functionality."""

    def test_semantic_version_creation(self):
        """Test creating semantic version objects."""
        version = SemanticVersion(1, 2, 3)
        assert version.major == 1
        assert version.minor == 2
        assert version.patch == 3
        assert str(version) == "1.2.3"

    def test_semantic_version_comparison(self):
        """Test semantic version comparison operations."""
        v1 = SemanticVersion(1, 0, 0)
        v2 = SemanticVersion(1, 1, 0)
        v3 = SemanticVersion(2, 0, 0)
        v4 = SemanticVersion(1, 0, 0)

        # Test equality
        assert v1 == v4
        assert v1 != v2

        # Test ordering
        assert v1 < v2
        assert v2 < v3
        assert v1 <= v2
        assert v2 <= v3
        assert v3 > v2
        assert v2 > v1
        assert v3 >= v2
        assert v2 >= v1

    def test_semantic_version_parsing(self):
        """Test parsing semantic version strings."""
        version = parse_semantic_version("1.2.3")
        assert version.major == 1
        assert version.minor == 2
        assert version.patch == 3

    def test_semantic_version_parsing_invalid(self):
        """Test parsing invalid semantic version strings."""
        with pytest.raises(ValueError):
            parse_semantic_version("1.2")

        with pytest.raises(ValueError):
            parse_semantic_version("1.2.3.4")

        with pytest.raises(ValueError):
            parse_semantic_version("v1.2.3")

        with pytest.raises(ValueError):
            parse_semantic_version("1.2.a")

    def test_validate_semantic_version(self):
        """Test semantic version validation."""
        assert validate_semantic_version("1.2.3") is True
        assert validate_semantic_version("0.0.1") is True
        assert validate_semantic_version("10.20.30") is True

        assert validate_semantic_version("1.2") is False
        assert validate_semantic_version("1.2.3.4") is False
        assert validate_semantic_version("v1.2.3") is False
        assert validate_semantic_version("1.2.a") is False


class TestVersionConstraints:
    """Test version constraint functionality."""

    def test_satisfies_version_constraint_exact(self):
        """Test exact version constraints."""
        version = SemanticVersion(1, 2, 3)

        assert satisfies_version_constraint(version, "1.2.3") is True
        assert satisfies_version_constraint(version, "==1.2.3") is True
        assert satisfies_version_constraint(version, "1.2.4") is False
        assert satisfies_version_constraint(version, "==1.2.4") is False

    def test_satisfies_version_constraint_range(self):
        """Test range version constraints."""
        version = SemanticVersion(1, 2, 3)

        assert satisfies_version_constraint(version, ">=1.0.0") is True
        assert satisfies_version_constraint(version, ">=1.2.3") is True
        assert satisfies_version_constraint(version, ">=1.2.4") is False

        assert satisfies_version_constraint(version, "<=2.0.0") is True
        assert satisfies_version_constraint(version, "<=1.2.3") is True
        assert satisfies_version_constraint(version, "<=1.2.2") is False

        assert satisfies_version_constraint(version, ">1.0.0") is True
        assert satisfies_version_constraint(version, ">1.2.3") is False

        assert satisfies_version_constraint(version, "<2.0.0") is True
        assert satisfies_version_constraint(version, "<1.2.3") is False

    def test_satisfies_version_constraint_compound(self):
        """Test compound version constraints."""
        version = SemanticVersion(1, 2, 3)

        assert satisfies_version_constraint(version, ">=1.0.0,<2.0.0") is True
        assert satisfies_version_constraint(version, ">=1.2.0,<=1.2.3") is True
        assert satisfies_version_constraint(version, ">=1.2.4,<2.0.0") is False
        assert satisfies_version_constraint(version, ">=1.0.0,<1.2.3") is False


class TestVersioningIntegration:
    """Test versioning integration with MLflow."""

    def test_log_transform_with_explicit_version(self, spark, mlflow_tracking):
        """Test logging transform with explicit version."""

        def sample_transform(df: DataFrame) -> DataFrame:
            return df.select("*")

        with mlflow.start_run() as run:
            log_transform_function(
                sample_transform,
                name="sample_transform",
                version="1.2.3",
            )

            # Verify version was logged correctly
            run_data = mlflow.get_run(run.info.run_id)
            assert run_data.data.tags["semantic_version"] == "1.2.3"
            assert run_data.data.params["major_version"] == "1"
            assert run_data.data.params["minor_version"] == "2"
            assert run_data.data.params["patch_version"] == "3"

    def test_log_transform_with_version_bump(self, spark, mlflow_tracking):
        """Test logging transform with version bump."""

        def sample_transform(df: DataFrame) -> DataFrame:
            return df.select("*")

        transform_name = "test_version_bump_transform"

        # Log first version
        with mlflow.start_run():
            log_transform_function(
                sample_transform,
                name=transform_name,
                version="1.0.0",
            )

        # Log second version with minor bump
        with mlflow.start_run() as run:
            log_transform_function(
                sample_transform,
                name=transform_name,
                version_bump="minor",
            )

            # Should be 1.1.0 (minor bump from 1.0.0)
            run_data = mlflow.get_run(run.info.run_id)
            assert run_data.data.tags["semantic_version"] == "1.1.0"

    def test_auto_version_generation(self, spark, mlflow_tracking):
        """Test automatic version generation."""

        def sample_transform(df: DataFrame) -> DataFrame:
            return df.select("*")

        transform_name = "test_auto_version_transform"

        # Log first version (should be 1.0.0)
        with mlflow.start_run() as run1:
            log_transform_function(sample_transform, name=transform_name)

            run_data = mlflow.get_run(run1.info.run_id)
            assert run_data.data.tags["semantic_version"] == "1.0.0"

        # Log second version (should auto-increment)
        with mlflow.start_run() as run2:
            log_transform_function(sample_transform, name=transform_name)

            run_data = mlflow.get_run(run2.info.run_id)
            # Should be 1.1.0 (minor bump as default)
            assert run_data.data.tags["semantic_version"] == "1.1.0"

    def test_get_latest_version(self, spark, mlflow_tracking):
        """Test getting latest version of a transform."""

        def sample_transform(df: DataFrame) -> DataFrame:
            return df.select("*")

        transform_name = "test_latest_version_transform"

        # Initially no versions
        latest = get_latest_version("nonexistent_transform")
        assert latest is None

        # Log multiple versions
        with mlflow.start_run():
            log_transform_function(
                sample_transform,
                name=transform_name,
                version="1.0.0",
            )

        with mlflow.start_run():
            log_transform_function(
                sample_transform,
                name=transform_name,
                version="1.2.0",
            )

        with mlflow.start_run():
            log_transform_function(
                sample_transform,
                name=transform_name,
                version="1.1.0",
            )

        # Should return the highest version
        latest = get_latest_version(transform_name)
        assert latest is not None
        assert latest == SemanticVersion(1, 2, 0)

    def test_find_transform_versions_with_constraints(self, spark, mlflow_tracking):
        """Test finding transforms with version constraints."""

        def sample_transform(df: DataFrame) -> DataFrame:
            return df.select("*")

        transform_name = "test_constraint_transform"

        # Log multiple versions
        with mlflow.start_run():
            log_transform_function(
                sample_transform,
                name=transform_name,
                version="1.0.0",
            )

        with mlflow.start_run():
            log_transform_function(
                sample_transform,
                name=transform_name,
                version="1.5.0",
            )

        with mlflow.start_run():
            log_transform_function(
                sample_transform,
                name=transform_name,
                version="2.0.0",
            )

        # Test version constraint filtering
        versions = find_transform_versions(
            name=transform_name,
            version_constraint=">=1.0.0,<2.0.0",
        )
        assert len(versions) == 2  # Should find 1.0.0 and 1.5.0

        # Test latest only
        versions = find_transform_versions(name=transform_name, latest_only=True)
        assert len(versions) == 1
        # Should be the latest version (2.0.0)
        if hasattr(versions, "iloc"):
            # It's a DataFrame
            version_tag = versions.iloc[0]["tags.semantic_version"]
        else:
            # It's a list or other structure
            version_tag = versions[0]["tags.semantic_version"]
        assert version_tag == "2.0.0"

    def test_interface_change_detection(self, spark, mlflow_tracking):
        """Test detection of interface changes for version bumping."""

        def transform_v1(df: DataFrame) -> DataFrame:
            """Original transform."""
            return df.select("*")

        def transform_v2(df: DataFrame, new_param: str = "default") -> DataFrame:
            """Transform with new optional parameter."""
            return df.select("*")

        def transform_v3(df: DataFrame, required_param: str) -> DataFrame:
            """Transform with new required parameter."""
            return df.select("*")

        transform_name = "test_interface_change_transform"

        # Log initial version
        with mlflow.start_run():
            log_transform_function(transform_v1, name=transform_name, version="1.0.0")

        # Log version with new optional parameter (should be minor bump)
        with mlflow.start_run() as run:
            log_transform_function(transform_v2, name=transform_name)

            run_data = mlflow.get_run(run.info.run_id)
            version = run_data.data.tags["semantic_version"]
            # Should be 1.1.0 (minor bump for compatible change)
            assert version == "1.1.0"

        # Log version with new required parameter (should be major bump)
        with mlflow.start_run() as run:
            log_transform_function(transform_v3, name=transform_name)

            run_data = mlflow.get_run(run.info.run_id)
            version = run_data.data.tags["semantic_version"]
            # Should be 2.0.0 (major bump for breaking change)
            assert version == "2.0.0"
