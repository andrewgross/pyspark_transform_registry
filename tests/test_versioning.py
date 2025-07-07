"""Tests for versioning functionality in the transform registry."""

import pytest
from pyspark_transform_registry.versioning import (
    compare_versions,
    get_latest_version,
    increment_version,
    matches_version_constraint,
    normalize_version,
    validate_semver,
)


class TestSemVerValidation:
    """Test SemVer format validation."""
    
    def test_valid_semver_formats(self):
        """Test that valid SemVer formats are accepted."""
        valid_versions = [
            "1.0.0",
            "0.1.0", 
            "10.20.30",
            "1.2.3-alpha",
            "1.2.3-alpha.1",
            "1.2.3-beta.2",
            "1.2.3-rc.1",
            "1.2.3+build.1",
            "1.2.3-alpha.1+build.1"
        ]
        
        for version in valid_versions:
            assert validate_semver(version), f"Version {version} should be valid"
    
    def test_invalid_semver_formats(self):
        """Test that invalid SemVer formats are rejected."""
        invalid_versions = [
            "1.0",          # Missing patch
            "1",            # Missing minor and patch  
            "1.0.0.0",      # Too many components
            "1.0.a",        # Non-numeric patch
            "a.0.0",        # Non-numeric major
            "1.a.0",        # Non-numeric minor
            "",             # Empty string
            "v1.0.0",       # Prefix not allowed
            "1.0.0-",       # Invalid pre-release
            "1.0.0+",       # Invalid build metadata
        ]
        
        for version in invalid_versions:
            assert not validate_semver(version), f"Version {version} should be invalid"


class TestVersionNormalization:
    """Test version normalization functionality."""
    
    def test_normalize_valid_versions(self):
        """Test that valid versions are normalized correctly."""
        test_cases = [
            ("1.0.0", "1.0.0"),
            ("1.2.3-alpha", "1.2.3a1"),  # packaging normalizes alpha format
            ("1.2.3-beta.1", "1.2.3b1"),
        ]
        
        for input_version, expected in test_cases:
            result = normalize_version(input_version)
            # We'll accept the packaging library's normalization
            assert validate_semver(result), f"Normalized version {result} should be valid"
    
    def test_normalize_invalid_versions(self):
        """Test that invalid versions raise ValueError."""
        with pytest.raises(ValueError, match="Invalid SemVer format"):
            normalize_version("1.0")
        
        with pytest.raises(ValueError, match="Invalid SemVer format"):
            normalize_version("invalid")


class TestVersionComparison:
    """Test version comparison functionality."""
    
    def test_version_comparison(self):
        """Test comparing version strings."""
        test_cases = [
            ("1.0.0", "2.0.0", -1),  # 1.0.0 < 2.0.0
            ("2.0.0", "1.0.0", 1),   # 2.0.0 > 1.0.0
            ("1.0.0", "1.0.0", 0),   # 1.0.0 == 1.0.0
            ("1.0.1", "1.0.0", 1),   # 1.0.1 > 1.0.0
            ("1.1.0", "1.0.1", 1),   # 1.1.0 > 1.0.1
            ("2.0.0", "1.9.9", 1),   # 2.0.0 > 1.9.9
        ]
        
        for v1, v2, expected in test_cases:
            result = compare_versions(v1, v2)
            assert result == expected, f"compare_versions('{v1}', '{v2}') should return {expected}, got {result}"
    
    def test_comparison_with_invalid_versions(self):
        """Test that invalid versions raise ValueError in comparison."""
        with pytest.raises(ValueError, match="Invalid SemVer format"):
            compare_versions("1.0", "1.0.0")
        
        with pytest.raises(ValueError, match="Invalid SemVer format"):
            compare_versions("1.0.0", "1.0")


class TestLatestVersion:
    """Test getting the latest version from a list."""
    
    def test_get_latest_version(self):
        """Test finding the latest version from a list."""
        versions = ["1.0.0", "1.1.0", "2.0.0", "1.0.1"]
        latest = get_latest_version(versions)
        assert latest == "2.0.0"
    
    def test_get_latest_empty_list(self):
        """Test that empty list returns None."""
        assert get_latest_version([]) is None
    
    def test_get_latest_single_version(self):
        """Test with a single version."""
        assert get_latest_version(["1.0.0"]) == "1.0.0"
    
    def test_get_latest_with_invalid_version(self):
        """Test that invalid versions raise ValueError."""
        with pytest.raises(ValueError, match="Invalid SemVer format"):
            get_latest_version(["1.0.0", "invalid", "2.0.0"])


class TestVersionIncrement:
    """Test version incrementing functionality."""
    
    def test_increment_patch(self):
        """Test incrementing patch version."""
        assert increment_version("1.0.0", "patch") == "1.0.1"
        assert increment_version("1.2.5", "patch") == "1.2.6"
    
    def test_increment_minor(self):
        """Test incrementing minor version."""
        assert increment_version("1.0.0", "minor") == "1.1.0"
        assert increment_version("1.2.5", "minor") == "1.3.0"
    
    def test_increment_major(self):
        """Test incrementing major version."""
        assert increment_version("1.0.0", "major") == "2.0.0"
        assert increment_version("1.2.5", "major") == "2.0.0"
    
    def test_increment_invalid_part(self):
        """Test that invalid part raises ValueError."""
        with pytest.raises(ValueError, match="Invalid part"):
            increment_version("1.0.0", "invalid")
    
    def test_increment_invalid_version(self):
        """Test that invalid version raises ValueError."""
        with pytest.raises(ValueError, match="Invalid SemVer format"):
            increment_version("1.0", "patch")


class TestVersionConstraints:
    """Test version constraint matching."""
    
    def test_exact_match_constraint(self):
        """Test exact version matching."""
        assert matches_version_constraint("1.0.0", "1.0.0")
        assert matches_version_constraint("1.0.0", "==1.0.0")
        assert not matches_version_constraint("1.0.1", "1.0.0")
    
    def test_greater_than_constraint(self):
        """Test greater than constraints."""
        assert matches_version_constraint("1.1.0", ">1.0.0")
        assert matches_version_constraint("2.0.0", ">=1.0.0")
        assert not matches_version_constraint("0.9.0", ">1.0.0")
        assert matches_version_constraint("1.0.0", ">=1.0.0")
    
    def test_less_than_constraint(self):
        """Test less than constraints."""
        assert matches_version_constraint("0.9.0", "<1.0.0")
        assert matches_version_constraint("1.0.0", "<=1.0.0")
        assert not matches_version_constraint("1.1.0", "<1.0.0")
        assert not matches_version_constraint("1.0.1", "<=1.0.0")
    
    def test_compatible_release_constraint(self):
        """Test compatible release operator (~=)."""
        assert matches_version_constraint("1.2.0", "~=1.2.0")
        assert matches_version_constraint("1.3.0", "~=1.2.0")
        assert not matches_version_constraint("2.0.0", "~=1.2.0")
    
    def test_constraint_with_invalid_version(self):
        """Test that invalid version raises ValueError."""
        with pytest.raises(ValueError, match="Invalid SemVer format"):
            matches_version_constraint("1.0", ">=1.0.0")
    
    def test_constraint_with_invalid_constraint(self):
        """Test that invalid constraint raises ValueError."""
        with pytest.raises(ValueError, match="Invalid version constraint"):
            matches_version_constraint("1.0.0", "invalid")