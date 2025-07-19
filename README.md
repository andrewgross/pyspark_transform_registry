# PySpark Transform Registry

A simplified library for registering and loading PySpark transform functions using MLflow's model registry.

## Installation

```bash
pip install pyspark-transform-registry
```

## Quick Start

### Register a Function

```python
from pyspark_transform_registry import register_function
from pyspark.sql import DataFrame
from pyspark.sql.functions import col

def clean_data(df: DataFrame) -> DataFrame:
    """Remove invalid records and standardize data."""
    return df.filter(col("amount") > 0).withColumn("status", lit("clean"))

# Register the function
model_uri = register_function(
    func=clean_data,
    name="analytics.etl.clean_data",
    description="Data cleaning transformation"
)
```

### Load and Use a Function

```python
from pyspark_transform_registry import load_function

# Load the registered function
clean_data_func = load_function("analytics.etl.clean_data")

# Use it on your data
result = clean_data_func(your_dataframe)
```

## Features

- **Simple API**: Just two main functions - `register_function()` and `load_function()`
- **Direct Registration**: Register functions directly from Python code
- **File-based Registration**: Load and register functions from Python files
- **Automatic Versioning**: Integer-based versioning with automatic incrementing
- **MLflow Integration**: Built on MLflow's model registry with automatic dependency inference
- **3-Part Naming**: Supports hierarchical naming (catalog.schema.table)

## Usage Examples

### Direct Function Registration

```python
from pyspark_transform_registry import register_function
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, when

def risk_scorer(df: DataFrame, threshold: float = 100.0) -> DataFrame:
    """Calculate risk scores based on amount."""
    return df.withColumn(
        "risk_score",
        when(col("amount") > threshold, "high").otherwise("low")
    )

# Register with metadata
register_function(
    func=risk_scorer,
    name="finance.scoring.risk_scorer",
    description="Risk scoring transformation",
    extra_pip_requirements=["numpy>=1.20.0"],
    tags={"team": "finance", "category": "scoring"}
)
```

### File-based Registration

```python
# transforms/data_processors.py
from pyspark.sql import DataFrame
from pyspark.sql.functions import col

def feature_engineer(df: DataFrame) -> DataFrame:
    """Create engineered features."""
    return df.withColumn("feature_1", col("amount") * 2)

def data_validator(df: DataFrame) -> DataFrame:
    """Validate data quality."""
    return df.filter(col("amount").isNotNull())
```

```python
# Register from file
register_function(
    file_path="transforms/data_processors.py",
    function_name="feature_engineer",
    name="ml.features.feature_engineer",
    description="Feature engineering pipeline"
)
```

### Loading and Versioning

```python
from pyspark_transform_registry import load_function, list_registered_functions

# Load latest version
transform = load_function("finance.scoring.risk_scorer")

# Load specific version
transform_v2 = load_function("finance.scoring.risk_scorer", version=2)

# List all registered functions
functions = list_registered_functions()
for func in functions:
    print(f"{func['name']} - Version {func['latest_version']}")
```

## API Reference

### `register_function()`

Register a PySpark transform function in MLflow's model registry.

**Parameters:**
- `func` (Callable, optional): The function to register (for direct registration)
- `name` (str): Model name for registry (supports 3-part naming)
- `file_path` (str, optional): Path to Python file containing the function
- `function_name` (str, optional): Name of function to extract from file
- `description` (str, optional): Model description
- `extra_pip_requirements` (list, optional): Additional pip requirements
- `tags` (dict, optional): Tags to attach to the registered model

**Returns:**
- `str`: Model URI of the registered model

### `load_function()`

Load a previously registered PySpark transform function.

**Parameters:**
- `name` (str): Model name in registry
- `version` (int or str, optional): Model version to load (defaults to latest)

**Returns:**
- `Callable`: The loaded transform function

### `list_registered_functions()`

List registered transform functions.

**Parameters:**
- `name_prefix` (str, optional): Optional prefix to filter model names

**Returns:**
- `list`: List of registered models with their metadata

## Requirements

- Python 3.8+
- PySpark 3.0+
- MLflow 2.0+

## Development

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run linting
ruff check --fix
ruff format
```

## License

MIT License
