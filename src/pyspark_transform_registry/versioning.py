"""
Version management utilities for PySpark Transform Registry.

Provides SemVer validation and version comparison functionality.
"""

import re
from typing import List, Optional

from packaging.version import Version, InvalidVersion


def validate_semver(version: str) -> bool:
    """
    Validate that a version string follows Semantic Versioning (SemVer) format.
    
    Args:
        version: Version string to validate (e.g., "1.2.3", "2.0.0-alpha.1")
        
    Returns:
        True if valid SemVer format, False otherwise
        
    Examples:
        >>> validate_semver("1.2.3")
        True
        >>> validate_semver("1.2")
        False
        >>> validate_semver("1.2.3-alpha.1")
        True
    """
    try:
        # Use packaging.version.Version which follows PEP 440 but is compatible with SemVer
        parsed = Version(version)
        
        # Additional check to ensure it's truly SemVer format (X.Y.Z)
        # SemVer requires at least major.minor.patch
        # Allow packaging library's normalization (e.g., "1.2.3a0" for "1.2.3-alpha")
        version_pattern = r'^\d+\.\d+\.\d+(?:(?:-[a-zA-Z0-9\-\.]+)|(?:[a-zA-Z]+\d*))?(?:\+[a-zA-Z0-9\-\.]+)?$'
        return bool(re.match(version_pattern, version))
    except InvalidVersion:
        return False


def normalize_version(version: str) -> str:
    """
    Normalize a version string to ensure consistent formatting.
    
    Args:
        version: Version string to normalize
        
    Returns:
        Normalized version string
        
    Raises:
        ValueError: If version is not valid SemVer format
    """
    if not validate_semver(version):
        raise ValueError(f"Invalid SemVer format: {version}")
    
    # Parse and reformat to ensure consistency
    parsed = Version(version)
    return str(parsed)


def compare_versions(version1: str, version2: str) -> int:
    """
    Compare two version strings.
    
    Args:
        version1: First version to compare
        version2: Second version to compare
        
    Returns:
        -1 if version1 < version2
         0 if version1 == version2  
         1 if version1 > version2
         
    Raises:
        ValueError: If either version is not valid SemVer format
    """
    if not validate_semver(version1):
        raise ValueError(f"Invalid SemVer format: {version1}")
    if not validate_semver(version2):
        raise ValueError(f"Invalid SemVer format: {version2}")
    
    v1 = Version(version1)
    v2 = Version(version2)
    
    if v1 < v2:
        return -1
    elif v1 > v2:
        return 1
    else:
        return 0


def get_latest_version(versions: List[str]) -> Optional[str]:
    """
    Get the latest version from a list of version strings.
    
    Args:
        versions: List of version strings
        
    Returns:
        Latest version string, or None if list is empty
        
    Raises:
        ValueError: If any version is not valid SemVer format
    """
    if not versions:
        return None
    
    valid_versions = []
    for v in versions:
        if not validate_semver(v):
            raise ValueError(f"Invalid SemVer format: {v}")
        valid_versions.append(Version(v))
    
    return str(max(valid_versions))


def increment_version(version: str, part: str = "patch") -> str:
    """
    Increment a version by the specified part.
    
    Args:
        version: Current version string
        part: Part to increment ("major", "minor", or "patch")
        
    Returns:
        New incremented version string
        
    Raises:
        ValueError: If version is not valid SemVer format or part is invalid
    """
    if not validate_semver(version):
        raise ValueError(f"Invalid SemVer format: {version}")
    
    if part not in ["major", "minor", "patch"]:
        raise ValueError(f"Invalid part: {part}. Must be 'major', 'minor', or 'patch'")
    
    parsed = Version(version)
    
    # Extract major, minor, patch from the version
    # Note: packaging.version doesn't have direct increment methods
    # so we'll parse manually
    version_parts = str(parsed).split('.')
    major = int(version_parts[0])
    minor = int(version_parts[1]) 
    patch = int(version_parts[2].split('-')[0].split('+')[0])  # Remove pre-release/build metadata
    
    if part == "major":
        return f"{major + 1}.0.0"
    elif part == "minor":
        return f"{major}.{minor + 1}.0"
    else:  # patch
        return f"{major}.{minor}.{patch + 1}"


def matches_version_constraint(version: str, constraint: str) -> bool:
    """
    Check if a version matches a version constraint.
    
    Args:
        version: Version string to check
        constraint: Version constraint (e.g., ">=1.0.0", "~=1.2.0", "==1.2.3")
        
    Returns:
        True if version matches constraint, False otherwise
        
    Raises:
        ValueError: If version or constraint is invalid
    """
    if not validate_semver(version):
        raise ValueError(f"Invalid SemVer format: {version}")
    
    try:
        parsed_version = Version(version)
        
        # Simple constraint parsing - extend this as needed
        if constraint.startswith(">="):
            min_version = Version(constraint[2:])
            return parsed_version >= min_version
        elif constraint.startswith("<="):
            max_version = Version(constraint[2:])
            return parsed_version <= max_version
        elif constraint.startswith(">"):
            min_version = Version(constraint[1:])
            return parsed_version > min_version
        elif constraint.startswith("<"):
            max_version = Version(constraint[1:])
            return parsed_version < max_version
        elif constraint.startswith("=="):
            exact_version = Version(constraint[2:])
            return parsed_version == exact_version
        elif constraint.startswith("~="):
            # Compatible release operator
            base_version = Version(constraint[2:])
            return parsed_version >= base_version and parsed_version.major == base_version.major
        else:
            # Default to exact match
            exact_version = Version(constraint)
            return parsed_version == exact_version
            
    except InvalidVersion:
        raise ValueError(f"Invalid version constraint: {constraint}")