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
from pyspark.sql.functions import col, lit

def clean_data(df: DataFrame) -> DataFrame:
    """Remove invalid records and standardize data."""
    return df.filter(col("amount") > 0).withColumn("status", lit("clean"))

# Register the function
logged_model = register_function(
    func=clean_data,
    name="analytics.etl.clean_data",
    description="Data cleaning transformation"
)
```

### Load and Use a Function

```python
from pyspark_transform_registry import load_function

# Load the registered function
clean_data_func = load_function("analytics.etl.clean_data", version=1)

# Use it on your data
result = clean_data_func(your_dataframe)
```

## Features

- **Simple API**: Just two main functions - `register_function()` and `load_function()`
- **Direct Registration**: Register functions directly from Python code
- **File-based Registration**: Load and register functions from Python files
- **Automatic Versioning**: Integer-based versioning with automatic incrementing
- **MLflow Integration**: Built on MLflow's model registry
- **Multi-Parameter Support**: Functions with additional parameters beyond the DataFrame

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

### Multi-Parameter Functions

```python
def filter_by_category(df: DataFrame, category: str, min_amount: float = 0.0) -> DataFrame:
    """Filter data by category and minimum amount."""
    return df.filter(
        (col("category") == category) &
        (col("amount") >= min_amount)
    )

# Register the function
register_function(
    func=filter_by_category,
    name="retail.filtering.filter_by_category"
)

# Load and use with parameters
filter_func = load_function("retail.filtering.filter_by_category", version=1)
electronics = filter_func(sample_df, category="electronics", min_amount=100.0)
```

### Source Code Inspection

```python
# Load a function
transform = load_function("retail.processing.process_orders", version=1)

# Get the original source code
source_code = transform.get_source()
print(source_code)  # Shows the original function definition

# Get the original function for inspection
original_func = transform.get_original_function()
print(f"Function name: {original_func.__name__}")
print(f"Docstring: {original_func.__doc__}")
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
- `infer_schema` (bool, optional): Whether to automatically infer schema constraints (default: False)
- `infer_requirements` (bool, optional): Whether to automatically infer pip requirements (default: False)

**Returns:**
- `ModelInfo`: The logged model information

### `load_function()`

Load a previously registered PySpark transform function.

**Parameters:**
- `name` (str): Model name in registry
- `version` (int or str): Model version to load (required)

**Returns:**
- `Callable`: The loaded transform function that supports:
  - Single param: `transform(df)`
  - Multi param: `transform(df, param1=value1, param2=value2)`
  - Source inspection: `transform.get_source()` - Returns the original function source code
  - Function access: `transform.get_original_function()` - Returns the unwrapped original function

### `get_latest_function_version()`

Get the latest version number of a function in the registry.

**Parameters:**
- `name` (str): Model name in registry

**Returns:**
- `str`: Latest version number of the function

## Model Discovery

To discover registered models, use MLflow's native model registry APIs:

```python
import mlflow
client = mlflow.tracking.MlflowClient()
models = client.list_registered_models()
for model in models:
    print(f"Model: {model.name}")
    for version in model.latest_versions:
        print(f"  Version: {version.version}")
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
