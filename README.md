# PySpark Transform Registry

A PySpark transform registry with MLflow integration for logging, versioning, and retrieving PySpark data transformation functions. This package enables reproducible data processing pipelines by persisting transform functions with metadata and allowing them to be reloaded later.

## Features

- **Function Persistence**: Log PySpark transform functions as artifacts in MLflow with complete source code and metadata
- **Reproducible Pipelines**: Reload previously logged transform functions for consistent data processing workflows
- **Version Management**: Track multiple versions of transform functions with semantic versioning and search capabilities
- **Type Safety**: Validate transform inputs using Python type hints to ensure data compatibility
- **Metadata Preservation**: Capture function signatures, parameters, return types, and docstrings
- **MLflow Integration**: Leverage MLflow's experiment tracking, artifact storage, and search capabilities

## Requirements

- Python 3.11+
- Java 17+ with security manager enabled
- PySpark < 4.0
- MLflow 2.22.0+

## Installation

### Using pip

```bash
pip install pyspark-transform-registry
```

### Using uv (recommended for development)

```bash
uv add pyspark-transform-registry
```

### Development Installation

```bash
git clone <repository-url>
cd pyspark_transform_registry
make install
# or
uv sync --extra dev
```

## Quick Start

### Basic Usage

```python
import mlflow
from pyspark.sql import SparkSession
from pyspark_transform_registry import (
    log_transform_function,
    load_transform_function,
    find_transform_versions
)

# Initialize Spark session
spark = SparkSession.builder.appName("TransformRegistry").getOrCreate()

# Set up MLflow tracking
mlflow.set_tracking_uri("your-mlflow-tracking-uri")
mlflow.set_experiment("transform-registry")

# Define a transform function
def add_profit_margin(df, margin_percent=0.15):
    """Add profit margin column to sales data."""
    return df.withColumn("profit_margin", df.price * margin_percent)

# Log the transform function
with mlflow.start_run():
    log_transform_function(
        transform_function=add_profit_margin,
        function_name="add_profit_margin",
        version="1.0.0",
        description="Adds profit margin calculation to sales data"
    )

# Load and use the transform function
loaded_transform = load_transform_function(
    function_name="add_profit_margin",
    version="1.0.0"
)

# Apply the loaded transform
df_with_margin = loaded_transform(sales_df, margin_percent=0.20)
```

### Version Management

```python
from pyspark_transform_registry import (
    find_transform_versions,
    get_latest_version,
    satisfies_version_constraint
)

# Find all versions of a transform
versions = find_transform_versions("add_profit_margin")
print(f"Available versions: {versions}")

# Get the latest version
latest = get_latest_version(versions)
print(f"Latest version: {latest}")

# Check version constraints
if satisfies_version_constraint("1.2.0", ">=1.0.0"):
    print("Version satisfies constraint")
```

### Input Validation

```python
from pyspark_transform_registry import validate_transform_input

# Validate input types before applying transform
def typed_transform(df: DataFrame, multiplier: float) -> DataFrame:
    """Transform with type hints for validation."""
    return df.withColumn("result", df.value * multiplier)

# Validate inputs match function signature
try:
    validate_transform_input(typed_transform, sales_df, multiplier=2.0)
    result = typed_transform(sales_df, multiplier=2.0)
except TypeError as e:
    print(f"Input validation failed: {e}")
```

## API Reference

### Core Functions

#### `log_transform_function(transform_function, function_name, version, description=None)`

Log a PySpark transform function to MLflow with metadata.

**Parameters:**
- `transform_function`: The PySpark transform function to log
- `function_name`: Name identifier for the function
- `version`: Semantic version string (e.g., "1.0.0")
- `description`: Optional description of the function

**Example:**
```python
log_transform_function(
    transform_function=my_transform,
    function_name="data_cleaner",
    version="1.0.0",
    description="Cleans and validates input data"
)
```

#### `load_transform_function(function_name, version)`

Load a previously logged transform function from MLflow.

**Parameters:**
- `function_name`: Name of the function to load
- `version`: Version string of the function to load

**Returns:**
- Callable transform function

**Example:**
```python
transform = load_transform_function("data_cleaner", "1.0.0")
cleaned_df = transform(raw_df)
```

#### `find_transform_versions(function_name, experiment_name=None)`

Find all versions of a transform function.

**Parameters:**
- `function_name`: Name of the function to search for
- `experiment_name`: Optional experiment name to search within

**Returns:**
- List of version strings

### Validation Functions

#### `validate_transform_input(transform_function, *args, **kwargs)`

Validate that input arguments match the function's type hints.

**Parameters:**
- `transform_function`: Function with type hints to validate against
- `*args, **kwargs`: Arguments to validate

**Raises:**
- `TypeError`: If arguments don't match expected types

### Version Management

#### `parse_semantic_version(version_string)`

Parse a semantic version string into components.

#### `validate_semantic_version(version_string)`

Validate that a version string follows semantic versioning.

#### `get_latest_version(versions)`

Get the latest version from a list of version strings.

#### `satisfies_version_constraint(version, constraint)`

Check if a version satisfies a version constraint.

## Development

### Setup

```bash
make install
uv run --extra dev pre-commit install
```

### Testing

```bash
# Run all tests
make test

# Run tests with verbose output
make test-verbose

# Run specific test file
uv run --extra dev pytest tests/test_transform_registry.py
```

### Code Quality

```bash
# Run all quality checks
make check

# Run linting
make lint

# Run formatting
make format
```

### Building

```bash
make build
```

## Environment Requirements

- **Java 17+**: Required for PySpark with security manager enabled
- **PySpark**: Configured for local[2] execution mode in tests
- **MLflow**: Uses temporary local tracking for test isolation

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests and quality checks (`make check`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## License

[Add your license information here]

## Support

[Add support information here]
