# SemVer Versioning Implementation for PySpark Transform Registry

## Overview

This document describes the implementation of Semantic Versioning (SemVer) support for the PySpark Transform Registry. The implementation allows users to store version identifiers with registered transforms following the SemVer format, with API capabilities to lookup versions without downloading the transformation code.

## Features Implemented

### 1. SemVer Version Validation
- **Function**: `validate_semver(version: str) -> bool`
- **Purpose**: Validates version strings against SemVer format (e.g., "1.2.3", "2.0.0-alpha.1")
- **Supports**: 
  - Standard SemVer format: X.Y.Z
  - Pre-release versions: X.Y.Z-alpha, X.Y.Z-beta.1, etc.
  - Build metadata: X.Y.Z+build.1
  - Python packaging library normalization (e.g., "1.2.3a0" for "1.2.3-alpha")

### 2. Enhanced Transform Logging
- **Function**: `log_transform_function()` - Enhanced with version support
- **New Parameters**:
  - `version: Optional[str] = None` - SemVer version string (defaults to "0.1.0")
  - `allow_version_overwrite: bool = False` - Allow overwriting existing name+version combinations
- **Features**:
  - Automatic version validation
  - Duplicate prevention (same name+version combination)
  - Version normalization for consistency
  - Backwards compatibility (version parameter is optional)

### 3. Version-Based Search and Retrieval
- **Enhanced Function**: `find_transform_versions()` - Added version filtering
- **New Parameters**:
  - `version: Optional[str] = None` - Filter by exact version
  - `version_constraint: Optional[str] = None` - Version constraint matching (e.g., ">=1.0.0")
- **New Functions**:
  - `get_transform_versions(name: str) -> List[str]` - Get all versions of a transform
  - `get_latest_transform_version(name: str) -> Optional[str]` - Get latest version
  - `load_transform_function_by_version(name: str, version: str = "latest")` - Load by version

### 4. Version Management Utilities
- **Version Comparison**: `compare_versions(v1: str, v2: str) -> int`
- **Version Incrementing**: `increment_version(version: str, part: str) -> str`
- **Latest Version Detection**: `get_latest_version(versions: List[str]) -> Optional[str]`
- **Version Constraint Matching**: `matches_version_constraint(version: str, constraint: str) -> bool`

## API Usage Examples

### Basic Version Logging
```python
from pyspark_transform_registry import log_transform_function

def my_transform(df: DataFrame) -> DataFrame:
    return df.select("*")

# Log with explicit version
with mlflow.start_run():
    log_transform_function(my_transform, "my_transform", version="1.2.3")

# Log with default version (0.1.0)
with mlflow.start_run():
    log_transform_function(my_transform, "my_transform")
```

### Version Discovery and Loading
```python
from pyspark_transform_registry import (
    get_transform_versions,
    get_latest_transform_version,
    load_transform_function_by_version
)

# Get all versions
versions = get_transform_versions("my_transform")
# Returns: ["1.0.0", "1.1.0", "2.0.0"]

# Get latest version
latest = get_latest_transform_version("my_transform")
# Returns: "2.0.0"

# Load specific version
transform_v1 = load_transform_function_by_version("my_transform", "1.0.0")

# Load latest version
transform_latest = load_transform_function_by_version("my_transform", "latest")
```

### Version Constraints and Search
```python
from pyspark_transform_registry import find_transform_versions

# Find specific version
runs = find_transform_versions(name="my_transform", version="1.2.3")

# Find versions matching constraint (future enhancement)
runs = find_transform_versions(name="my_transform", version_constraint=">=1.0.0")
```

## Storage Implementation

### MLflow Integration
- **Version Storage**: Versions are stored as MLflow parameters (`params.version`)
- **Searchability**: Versions can be searched without downloading transform code
- **Metadata**: All version information is available via MLflow run metadata
- **Backwards Compatibility**: Existing transforms without versions continue to work

### Data Structure
```python
# MLflow Run Parameters
{
    "transform_name": "my_transform",
    "version": "1.2.3",
    "return_type": "pyspark.sql.DataFrame",
    "param_info": "[{...}]",
    "docstring": "Function documentation"
}

# MLflow Run Tags
{
    "transform_type": "pyspark"
}

# MLflow Run Metrics
{
    "timestamp": 1640995200.0
}
```

## Error Handling

### Version Validation Errors
```python
# Invalid SemVer format
log_transform_function(func, "name", version="1.0")  # Raises ValueError

# Duplicate version prevention
log_transform_function(func, "name", version="1.0.0")  # First time: OK
log_transform_function(func, "name", version="1.0.0")  # Second time: Raises ValueError

# Override duplicate prevention
log_transform_function(func, "name", version="1.0.0", allow_version_overwrite=True)  # OK
```

### Retrieval Errors
```python
# Nonexistent version
load_transform_function_by_version("name", "99.0.0")  # Raises ValueError

# Nonexistent transform
get_transform_versions("nonexistent")  # Raises ValueError
get_latest_transform_version("nonexistent")  # Returns None
```

## Dependencies Added

- **packaging>=21.0**: For robust SemVer parsing and comparison
  - Handles version normalization
  - Provides reliable version comparison
  - Compatible with Python packaging standards

## Testing

### Unit Tests (`tests/test_versioning.py`)
- SemVer validation for various version formats
- Version normalization and comparison
- Version incrementing functionality
- Version constraint matching
- Error handling for invalid inputs

### Integration Tests (`tests/test_versioning_integration.py`)
- End-to-end workflow with MLflow backend
- Version logging, storage, and retrieval
- Multi-version scenarios
- Search functionality validation
- Error handling in real scenarios

## Backwards Compatibility

- **Existing Code**: All existing code continues to work unchanged
- **Default Behavior**: Transforms without explicit version get "0.1.0"
- **API Extensions**: New parameters are optional with sensible defaults
- **Migration Path**: Existing transforms can be re-logged with explicit versions

## Performance Considerations

- **Version Queries**: Metadata-only queries don't download transform code
- **Search Efficiency**: Version filtering uses MLflow's built-in parameter search
- **Storage Overhead**: Minimal - only adds one parameter per transform
- **Memory Usage**: Version lists are lightweight string collections

## Future Enhancements

1. **Advanced Version Constraints**: 
   - Support for complex constraint expressions
   - Version range specifications
   - Dependency resolution

2. **Version History**:
   - Track version lineage and dependencies
   - Change logs between versions
   - Automated version suggestions

3. **Version Tags**:
   - Support for version aliases (e.g., "stable", "latest-beta")
   - Environment-specific version pinning

4. **Migration Tools**:
   - Batch version assignment for existing transforms
   - Version import/export utilities

## Implementation Notes

### Design Decisions
1. **Optional Version Parameter**: Maintains backwards compatibility while encouraging versioning
2. **Default Version "0.1.0"**: Follows semantic versioning conventions for initial releases
3. **Strict SemVer Validation**: Ensures consistent version format across the system
4. **MLflow Parameter Storage**: Leverages existing metadata system for efficient queries
5. **Duplicate Prevention**: Prevents accidental overwrites while allowing explicit overrides

### Limitations
1. **Version Constraint Filtering**: Currently done post-search due to MLflow query limitations
2. **Type Annotations**: Some MLflow return type mismatches due to library evolution
3. **Pre-release Normalization**: Packaging library normalization may differ from pure SemVer

The implementation successfully provides comprehensive SemVer versioning support while maintaining full backwards compatibility and enabling efficient version-based API operations without requiring transform download.