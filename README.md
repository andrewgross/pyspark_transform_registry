# PySpark Transform Registry

A PySpark transform registry with MLflow integration that provides semantic versioning support for managing data transformation functions.

## Features

- **Transform Storage**: Store PySpark transformation functions with metadata in MLflow
- **Semantic Versioning**: Full SemVer support for versioning transforms (1.2.3, 2.0.0-alpha, etc.)
- **Version Management**: Discover, compare, and load specific versions of transforms
- **Metadata Rich**: Automatic extraction of function signatures, documentation, and type hints
- **Search & Discovery**: Find transforms by name, version, return type, and constraints
- **API-First**: Query version information without downloading transform code

## Installation

```bash
pip install pyspark-transform-registry
```

### Requirements

- Python 3.11+
- PySpark <4.0
- MLflow >=2.22.0
- Java 17+ with security manager enabled

## Quick Start

### Basic Usage

```python
import mlflow
from pyspark.sql import DataFrame
from pyspark_transform_registry import log_transform_function, load_transform_function_by_version

# Define a transform function
def clean_data(df: DataFrame) -> DataFrame:
    """Remove null values and standardize column names."""
    return df.dropna().select([col(c).alias(c.lower()) for c in df.columns])

# Log the transform with a version
with mlflow.start_run():
    log_transform_function(
        clean_data, 
        name="data_cleaner",
        version="1.0.0"
    )

# Load and use the transform later
cleaner = load_transform_function_by_version("data_cleaner", "1.0.0")
cleaned_df = cleaner(raw_df)
```

### Version Management

```python
from pyspark_transform_registry import (
    get_transform_versions,
    get_latest_transform_version,
    find_transform_versions
)

# Discover all versions of a transform
versions = get_transform_versions("data_cleaner")
# Returns: ["1.0.0", "1.1.0", "2.0.0"]

# Get the latest version
latest = get_latest_transform_version("data_cleaner")
# Returns: "2.0.0"

# Search for specific versions
runs = find_transform_versions(name="data_cleaner", version="1.0.0")

# Load the latest version automatically
latest_cleaner = load_transform_function_by_version("data_cleaner", "latest")
```

### Version Validation

```python
from pyspark_transform_registry import validate_semver, increment_version

# Validate version format
assert validate_semver("1.2.3") == True
assert validate_semver("1.0") == False

# Increment versions
next_patch = increment_version("1.2.3", "patch")  # "1.2.4"
next_minor = increment_version("1.2.3", "minor")  # "1.3.0"
next_major = increment_version("1.2.3", "major")  # "2.0.0"
```

## API Reference

### Core Functions

#### `log_transform_function(func, name, version=None, ...)`
Store a transform function with metadata and version information.

**Parameters:**
- `func`: The PySpark transform function to store
- `name`: Unique name for the transform
- `version`: SemVer version string (defaults to "0.1.0")
- `allow_version_overwrite`: Allow overwriting existing name+version combinations

#### `load_transform_function_by_version(name, version="latest")`
Load a stored transform function by name and version.

**Parameters:**
- `name`: Name of the transform to load
- `version`: Version to load ("latest" or specific version like "1.2.3")

#### `get_transform_versions(name)`
Get all available versions of a transform.

#### `get_latest_transform_version(name)`
Get the latest version of a transform.

#### `find_transform_versions(name=None, version=None, return_type=None)`
Search for transforms with optional filtering.

### Version Utilities

#### `validate_semver(version)`
Validate if a version string follows SemVer format.

#### `compare_versions(v1, v2)`
Compare two version strings (-1, 0, or 1).

#### `increment_version(version, part)`
Increment a version by major, minor, or patch.

#### `matches_version_constraint(version, constraint)`
Check if a version matches a constraint (e.g., ">=1.0.0").

## Storage & Metadata

### MLflow Integration

The registry uses MLflow to store transforms with rich metadata:

```python
# Stored as MLflow parameters (searchable without download)
{
    "transform_name": "data_cleaner",
    "version": "1.2.3",
    "return_type": "pyspark.sql.DataFrame",
    "param_info": "[{...}]",
    "docstring": "Function documentation"
}
```

### Version Storage

- **Format**: Semantic Versioning (SemVer 2.0.0)
- **Examples**: `1.0.0`, `2.1.3-alpha.1`, `1.0.0+build.123`
- **Storage**: MLflow parameters (efficient metadata queries)
- **Search**: Query versions without downloading transform code

## Error Handling

```python
# Version validation
try:
    log_transform_function(func, "name", version="1.0")  # Invalid format
except ValueError as e:
    print("Invalid SemVer format")

# Duplicate prevention
try:
    log_transform_function(func, "name", version="1.0.0")  # Second time
except ValueError as e:
    print("Version already exists")

# Override duplicates
log_transform_function(func, "name", version="1.0.0", allow_version_overwrite=True)

# Handle missing versions
try:
    load_transform_function_by_version("name", "99.0.0")
except ValueError as e:
    print("Version not found")
```

## Examples

See `examples/versioning_example.py` for a comprehensive demonstration of all versioning features.

## Testing

Run the test suite:

```bash
# Unit tests for versioning utilities
pytest tests/test_versioning.py

# Integration tests with MLflow backend
pytest tests/test_versioning_integration.py

# All tests
pytest
```

## Advanced Usage

### Version Constraints

```python
# Find versions matching constraints
from pyspark_transform_registry import matches_version_constraint

version = "1.2.3"
print(matches_version_constraint(version, ">=1.0.0"))  # True
print(matches_version_constraint(version, "~=1.2.0"))   # True (compatible release)
print(matches_version_constraint(version, "<1.0.0"))    # False
```

### Batch Operations

```python
# Get version history
versions = get_transform_versions("my_transform")
for version in versions:
    print(f"Version {version} available")

# Load multiple versions for comparison
v1_func = load_transform_function_by_version("transform", "1.0.0")
v2_func = load_transform_function_by_version("transform", "2.0.0")
```

## Backwards Compatibility

- All existing code continues to work unchanged
- New version parameter is optional (defaults to "0.1.0")
- Existing transforms can be re-logged with explicit versions
- No breaking changes to existing APIs

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

[License information]

## Documentation

- [Versioning Implementation Details](VERSIONING_IMPLEMENTATION.md)
- [API Reference](docs/api.md)
- [Examples](examples/)