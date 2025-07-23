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
- **MLflow Integration**: Built on MLflow's model registry with automatic dependency inference
- **3-Part Naming**: Supports hierarchical naming (catalog.schema.table)
- **Runtime Validation**: Automatic schema inference and DataFrame validation before execution
- **Type Safety**: Validate input DataFrames against inferred schema constraints
- **Flexible Validation**: Support for both strict and permissive validation modes
- **Source Code Inspection**: Access original function source code and metadata for debugging

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
from pyspark_transform_registry import load_function

# Load latest version
transform = load_function("finance.scoring.risk_scorer", version=1)

# Load specific version
transform_v2 = load_function("finance.scoring.risk_scorer", version=2)

# Use MLflow's native model registry APIs to discover models
# See MLflow documentation for model discovery patterns
```

### Runtime Validation

The registry automatically infers schema constraints from your functions and validates input DataFrames before execution.

```python
from pyspark_transform_registry import register_function, load_function
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, lit

def process_orders(df: DataFrame) -> DataFrame:
    """Process order data with specific column requirements."""
    return (df
        .filter(col("amount") > 0)
        .withColumn("processed", lit(True))
        .select("order_id", "customer_id", "amount", "processed")
    )

# Register with automatic schema inference
register_function(
    func=process_orders,
    name="retail.processing.process_orders",
    infer_schema=True  # Default: True
)

# Load with validation enabled (default)
transform = load_function("retail.processing.process_orders", version=1)

# This will validate the DataFrame structure before processing
result = transform(orders_df)  # Validates: order_id, customer_id, amount columns exist

# Load with validation disabled
transform_no_validation = load_function(
    "retail.processing.process_orders",
    version=1,
    validate_input=False
)

# Load with strict validation (warnings become errors)
transform_strict = load_function(
    "retail.processing.process_orders",
    version=1,
    strict_validation=True
)
```

### Multi-Parameter Functions with Validation

```python
def filter_by_category(df: DataFrame, category: str, min_amount: float = 0.0) -> DataFrame:
    """Filter data by category and minimum amount."""
    return df.filter(
        (col("category") == category) &
        (col("amount") >= min_amount)
    )

sample_df = spark.createDataFrame([
    ("electronics", 100.0, "order_1"),
    ("books", 25.0, "order_2")
], ["category", "amount", "order_id"])

register_function(
    func=filter_by_category,
    name="retail.filtering.filter_by_category",
)

# Load and use with parameters
filter_func = load_function("retail.filtering.filter_by_category", version=1)

# Use with validation - validates DataFrame structure before filtering
electronics = filter_func(sample_df, params={"category": "electronics", "min_amount": 100.0})
```

### Source Code Inspection

The loaded functions provide access to the original transform source code for debugging and understanding:

```python
# Load a function
transform = load_function("retail.processing.process_orders", version=1)

# Get the original source code
source_code = transform.get_source()
print(source_code)  # Shows the original function definition

# Get the original function for advanced inspection
original_func = transform.get_original_function()
print(f"Function name: {original_func.__name__}")
print(f"Docstring: {original_func.__doc__}")

# Use inspect on the original function
import inspect
signature = inspect.signature(original_func)
print(f"Signature: {signature}")

# Note: inspect.getsource(transform) shows wrapper code
# transform.get_source() shows the original function code
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
- `infer_schema` (bool, optional): Whether to automatically infer schema constraints (default: True)
- `schema_constraint` (PartialSchemaConstraint, optional): Pre-computed schema constraint

**Returns:**
- `str`: Model URI of the registered model

### `load_function()`

Load a previously registered PySpark transform function with optional validation.

**Parameters:**
- `name` (str): Model name in registry
- `version` (int or str): Model version to load (required)
- `validate_input` (bool, optional): Whether to validate input DataFrames against stored schema constraints (default: True)
- `strict_validation` (bool, optional): If True, treat validation warnings as errors (default: False)

**Returns:**
- `Callable`: The loaded transform function that supports both single and multi-parameter usage:
  - Single param: `transform(df)`
  - Multi param: `transform(df, params={'param1': value1, 'param2': value2})`
  - Source inspection: `transform.get_source()` - Returns the original function source code
  - Function access: `transform.get_original_function()` - Returns the unwrapped original function

### Model Discovery

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

- Python 3.11+
- PySpark 3.0+
- MLflow 2.22+

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
