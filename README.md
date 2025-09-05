# PySpark Transform Registry

A simplified library for registering and loading PySpark transform functions using MLflow's model registry.

## Installation

```bash
pip install pyspark-transform-registry
```

```bash
uv add pyspark-transform-registry
```

## Quick Start

### Register a Function

```python
from pyspark_transform_registry import register_transform
from pyspark.sql import DataFrame
import pyspark.sql.functions as F

def clean_data(df: DataFrame) -> DataFrame:
    """Remove invalid records and standardize data."""
    return df.filter(F.col("amount") > 0).withColumn("status", F.lit("clean"))

# Register the transform
logged_model = register_transform(
    func=clean_data,
    name="analytics.etl.clean_data",
    description="Data cleaning transformation"
)
```

### Load and Use a Transform

```python
from pyspark_transform_registry import load_transform, load_transform_uri

# Load the registered transform
clean_data_func = load_transform("analytics.etl.clean_data", version=1)

# Or
clean_data_func = load_transform_uri("transforms:/analytics.etl.clean_data/1")

# Use it on your data
result = clean_data_func(your_dataframe)
```

## Features

- **Simple API**: Just two main functions - `register_transform()` and `load_transform()`
- **Direct Registration**: Register transforms directly from Python code
- **File-based Registration**: Load and register transforms from Python files
- **Automatic Versioning**: Integer-based versioning with automatic incrementing
- **MLflow Integration**: Built on MLflow's model registry

## Usage Examples

### Direct Transform Registration

```python
from pyspark_transform_registry import register_transform
from pyspark.sql import DataFrame
import pyspark.sql.functions as F

def risk_scorer(df: DataFrame, threshold: float = 100.0) -> DataFrame:
    """Calculate risk scores based on amount."""
    return df.withColumn(
        "risk_score",
        F.when(F.col("amount") > threshold, "high").otherwise("low")
    )

# Register with metadata
register_transform(
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
import pyspark.sql.functions as F

def feature_engineer(df: DataFrame) -> DataFrame:
    """Create engineered features."""
    return df.withColumn("feature_1", col("amount") * 2)
```

```python
# Register from file
register_transform(
    file_path="transforms/data_processors.py",
    function_name="feature_engineer",
    name="ml.features.feature_engineer",
    description="Feature engineering pipeline"
)
```

### Source Code Inspection

```python
# Load a transform
transform = load_transform("retail.processing.process_orders", version=1)

# Get the original source code
source_code = transform.get_source()
print(source_code)  # Shows the original function definition

# Get the original function for inspection
original_func = transform.get_original_function()
print(f"Function name: {original_func.__name__}")
print(f"Docstring: {original_func.__doc__}")
```

## Requirements

- Python 3.9+
- PySpark 3.0+
- MLflow 3.0+

## Development

```bash
# Install development dependencies
make install

# Run tests
make test

# Run linting and formatting
make check
```

## License

MIT License
