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
- **Runtime Validation**: Automatic schema inference and DataFrame validation before execution
- **Type Safety**: Validate input DataFrames against inferred schema constraints
- **Flexible Validation**: Support for both strict and permissive validation modes

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
transform = load_function("retail.processing.process_orders")

# This will validate the DataFrame structure before processing
result = transform(orders_df)  # Validates: order_id, customer_id, amount columns exist

# Load with validation disabled
transform_no_validation = load_function(
    "retail.processing.process_orders",
    validate_input=False
)

# Load with strict validation (warnings become errors)
transform_strict = load_function(
    "retail.processing.process_orders",
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

# Register with example for signature inference
sample_df = spark.createDataFrame([
    ("electronics", 100.0, "order_1"),
    ("books", 25.0, "order_2")
], ["category", "amount", "order_id"])

register_function(
    func=filter_by_category,
    name="retail.filtering.filter_by_category",
    input_example=sample_df,
    example_params={"category": "electronics", "min_amount": 50.0}
)

# Load and use with parameters
filter_func = load_function("retail.filtering.filter_by_category")

# Use with validation - validates DataFrame structure before filtering
electronics = filter_func(orders_df, params={"category": "electronics", "min_amount": 100.0})
```

## API Reference

### `register_function()`

Register a PySpark transform function in MLflow's model registry.

**Parameters:**
- `func` (Callable, optional): The function to register (for direct registration)
- `name` (str): Model name for registry (supports 3-part naming)
- `file_path` (str, optional): Path to Python file containing the function
- `function_name` (str, optional): Name of function to extract from file
- `input_example` (DataFrame, optional): Sample input DataFrame for signature inference
- `example_params` (dict, optional): Example parameters for multi-parameter functions
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
- `version` (int or str, optional): Model version to load (defaults to latest)
- `validate_input` (bool, optional): Whether to validate input DataFrames against stored schema constraints (default: True)
- `strict_validation` (bool, optional): If True, treat validation warnings as errors (default: False)

**Returns:**
- `Callable`: The loaded transform function that supports both single and multi-parameter usage:
  - Single param: `transform(df)`
  - Multi param: `transform(df, params={'param1': value1, 'param2': value2})`

### `list_registered_functions()`

List registered transform functions.

**Parameters:**
- `name_prefix` (str, optional): Optional prefix to filter model names

**Returns:**
- `list`: List of registered models with their metadata

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
