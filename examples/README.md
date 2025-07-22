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
# Log a simple transform
register_function(add_sales_tax, "add_sales_tax")

# Apply with different parameters
result = add_sales_tax(df, tax_rate=0.10)

# Reload and reuse
transform = load_function("add_sales_tax")
result = transform(df, tax_rate=0.08)
```

### 2. Intermediate Transforms

**Functions**: `customer_segmentation()`, `clean_and_validate_data()`

Multi-step transformations with business logic:
- Customer segmentation with conditional logic
- Data validation and cleaning
- Multiple column operations in one transform

```python
# Log intermediate transform
register_function(customer_segmentation, "customer_segmentation")

# Apply segmentation
segmented_df = customer_segmentation(df)

# Find all available transforms
transforms = list_registered_functions()
```

### 3. Complex Transforms

**Functions**: `advanced_feature_engineering()`, `ml_preprocessing_pipeline()`

Advanced transformations with dependencies:
- Window functions and aggregations
- Feature engineering for ML
- Multi-stage preprocessing pipelines
- Custom business logic

```python
# Log complex transform
register_function(advanced_feature_engineering, "advanced_features")

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

**Functions**: `example_5_version_management()`

Shows how to:
- Create multiple versions of transforms
- Use semantic versioning
- Apply version constraints
- Compare different versions

```python
# Log different versions
register_function(calculate_discount_v1, "calculate_discount", version="1.0.0")
register_function(calculate_discount_v2, "calculate_discount", version="2.0.0")

# Load specific version ranges
v1_transform = load_function("calculate_discount", version_constraint=">=1.0.0,<2.0.0")
v2_transform = load_function("calculate_discount", version_constraint=">=2.0.0")
```

## Key Patterns Demonstrated

### Transform Registration
```python
# Basic registration
register_function(my_transform, "my_transform")

# With versioning
register_function(my_transform, "my_transform", version="1.0.0")

# With examples for schema inference
register_function(my_transform, "my_transform",
                      input_example=df, output_example=result_df)
```

### Transform Loading
```python
# Load by name (latest version)
transform = load_function("my_transform")

# Load with version constraints
transform = load_function("my_transform", version_constraint=">=1.0.0,<2.0.0")

# Load specific version
transform = load_transform_function("my_transform", version="1")
```

### Transform Discovery
```python
# Find all transforms
all_transforms = list_registered_functions()

# Find by name
name_transforms = list_registered_functions(name="my_transform")

# Find with version constraints
constrained_transforms = list_registered_functions(
    name="my_transform",
    version_constraint=">=1.0.0"
)
```

### Transform Chaining
```python
# Chain transforms in sequence
step1_transform = load_function("data_cleaning")
step2_transform = load_function("feature_engineering")

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
