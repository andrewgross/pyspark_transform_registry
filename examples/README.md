# PySpark Transform Registry Examples

This directory contains comprehensive examples showing how to use the PySpark Transform Registry with varying complexity levels.

## Overview

The examples demonstrate:
1. **Simple transforms** - Basic column operations
2. **Intermediate transforms** - Multiple operations and business logic
3. **Complex transforms** - Advanced feature engineering and ML preprocessing
4. **Workflow examples** - Chaining and reusing transforms
5. **Version management** - Managing multiple versions of transforms

## Running the Examples

```bash
# Install dependencies
uv sync --extra dev

# Run all examples
uv run --extra dev python examples/transform_examples.py

# Or run individual examples in a Python session
python -c "
from examples.transform_examples import *
example_1_simple_workflow()
"
```

## Example Categories

### 1. Simple Transforms

**Functions**: `add_sales_tax()`, `normalize_text()`

Simple column operations that demonstrate basic usage patterns:
- Adding calculated columns
- Text normalization
- Basic parameter handling

```python
# Register a simple transform
register_function(func=add_sales_tax, name="add_sales_tax")

# Apply with different parameters
result = add_sales_tax(df, tax_rate=0.10)

# Reload and reuse
transform = load_function("add_sales_tax", version=1)
result = transform(df, params={"tax_rate": 0.08})
```

### 2. Intermediate Transforms

**Functions**: `customer_segmentation()`, `clean_and_validate_data()`

Multi-step transformations with business logic:
- Customer segmentation with conditional logic
- Data validation and cleaning
- Multiple column operations in one transform

```python
# Register intermediate transform
register_function(func=customer_segmentation, name="customer_segmentation")

# Apply segmentation
segmented_df = customer_segmentation(df)

# Use MLflow's model registry to discover models
import mlflow
client = mlflow.tracking.MlflowClient()
transforms = client.list_registered_models()
```

### 3. Complex Transforms

**Functions**: `advanced_feature_engineering()`, `ml_preprocessing_pipeline()`

Advanced transformations with dependencies:
- Window functions and aggregations
- Feature engineering for ML
- Multi-stage preprocessing pipelines
- Custom business logic

```python
# Register complex transform
register_function(func=advanced_feature_engineering, name="advanced_features")

# Apply feature engineering
featured_df = advanced_feature_engineering(df)
```

### 4. Workflow Examples

**Functions**: `example_1_simple_workflow()` through `example_5_version_management()`

Complete workflows showing:
- **Simple workflow**: Basic save/load/apply pattern
- **Intermediate workflow**: Business logic with search functionality
- **Complex workflow**: Multi-stage feature engineering
- **ML pipeline**: Full data cleaning and preprocessing pipeline
- **Version management**: Multiple versions with constraints

```python
# Run complete ML pipeline
example_4_full_ml_pipeline()

# Demonstrates:
# 1. Data cleaning transform
# 2. ML preprocessing transform
# 3. Chaining transforms
# 4. Reloading and reusing transforms
```

### 5. Version Management



```python
# Register different versions of the same function
model_uri_v1 = register_function(func=calculate_discount_v1, name="calculate_discount")
# model:/calculate_discount/1

model_uri_v2 = register_function(func=calculate_discount_v2, name="calculate_discount")
# model:/calculate_discount/2

# Load specific versions
v1_func = load_function("calculate_discount", version=1)
v2_func = load_function("calculate_discount", version=2)
```

## Key Patterns Demonstrated

### Transform Registration
```python
# Basic registration
register_function(func=my_transform, name="my_transform")

# With schema inference
register_function(func=my_transform, name="my_transform", input_example=df)

# With metadata
register_function(
    func=my_transform,
    name="my_transform",
    input_example=df,
    description="My transform function"
)
```

### Transform Loading
```python
# Load by name
transform = load_function("my_transform", version=1)

```

### Transform Discovery
```python
# Discover transforms using MLflow
import mlflow
client = mlflow.tracking.MlflowClient()
all_transforms = client.list_registered_models()

# Get specific model info
model = client.get_registered_model("my_transform")
versions = model.latest_versions

# List all versions of a model
versions = client.get_latest_versions("my_transform")
```

### Transform Chaining
```python
# Chain transforms in sequence
step1_transform = load_function("data_cleaning", version=1)
step2_transform = load_function("feature_engineering", version=1)

# Apply pipeline
result = step2_transform(step1_transform(raw_df))
```

## Best Practices Shown

1. **Type Annotations**: All transforms use proper type hints
2. **Documentation**: Functions include comprehensive docstrings
3. **Error Handling**: Robust error handling in complex transforms
4. **Modularity**: Transforms are focused and reusable
5. **Testing**: Each example includes validation of results
6. **Versioning**: Proper semantic versioning practices
7. **Metadata**: Rich metadata for discoverability

## Sample Data Structures

The examples use these common data structures:

```python
# Sales data
sales_df = [(1, "apple", 2.50), (2, "banana", 1.20)]
Schema: [id: int, product: string, price: double]

# Customer data
customer_df = [(1, 1500, 8), (2, 750, 4)]
Schema: [customer_id: int, total_spent: double, purchase_frequency: int]

# Transaction data
transaction_df = [(1, 1, "2023-01-01", 100.0), (2, 1, "2023-01-15", 250.0)]
Schema: [transaction_id: int, customer_id: int, transaction_date: string, amount: double]

# ML data
ml_df = [(1, 25, 45000, 650, "john@email.com", "123-456-7890", "John Doe")]
Schema: [id: int, age: int, income: double, credit_score: int, email: string, phone: string, name: string]
```

## Dependencies

The examples require:
- PySpark 3.x
- MLflow 2.x
- The pyspark_transform_registry package

All dependencies are managed through the project's `pyproject.toml` file.
