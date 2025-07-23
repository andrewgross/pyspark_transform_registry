# TODO: Type System Improvements

## Root Cause Analysis - Type Inference Issues

### Problem
The static analysis system has several critical issues with type inference that cause it to fall back to unreliable column name pattern matching:

1. **Data Structure Mismatch**:
   - `TypeInferenceEngine.type_mappings` returns `dict[str, TypeInference]`
   - `ConstraintGenerator._get_column_type()` expects `type_info` to have an "inferred_types" key structure
   - Code tries to access `type_info.get("inferred_types", {})` but `type_info` is the direct `type_mappings` dict

2. **Empty Type Info**:
   - The `type_info` parameter is consistently empty due to the data structure mismatch
   - Located in `analyzer.py:57` where we pass `type_engine.type_mappings` as `type_info`
   - But `schema_inference.py:267` expects a different structure

3. **Unacceptable Fallback**:
   - Current `_get_column_type()` function in `schema_inference.py:265-289` guesses types based on column name patterns
   - Examples: columns with "id" become "string", "amount" becomes "double", etc.
   - This is unreliable and should be eliminated

### Files Affected
- `src/pyspark_transform_registry/static_analysis/schema_inference.py` (lines 265-289)
- `src/pyspark_transform_registry/static_analysis/analyzer.py` (line 57)
- `src/pyspark_transform_registry/static_analysis/type_inference.py` (TypeInference class)

## Planned Solutions

### High Priority
- [ ] **Fix data structure mismatch between TypeInferenceEngine and ConstraintGenerator**
  - Align the data structure expectations between `analyzer.py:57` and `schema_inference.py:267`
  - Either change how `type_engine.type_mappings` is passed or how it's consumed

- [ ] **Replace _get_column_type with proper type inference**
  - Remove the name-pattern guessing logic in `schema_inference.py:265-289`
  - Use actual type information from the TypeInferenceEngine when available
  - Implement proper fallback that doesn't rely on naming patterns

### Medium Priority
- [ ] **Add UnknownType class for underspecified column types**
  - Create a proper `UnknownType` class to represent columns where type cannot be determined
  - Include confidence levels, constraints, and context information
  - Better than guessing types or returning generic "string"

- [ ] **Create robust fallback when type information is missing**
  - Design a system that gracefully handles missing type information
  - Provide clear warnings when types cannot be determined
  - Avoid making assumptions about column types based on names

## Implementation Notes

### Current Behavior
```python
# In schema_inference.py:267
inferred_types = type_info.get("inferred_types", {})  # Always empty!

# In schema_inference.py:275-288
# Falls back to name-based guessing:
if any(keyword in name_lower for keyword in ["id", "key"]):
    return "string"
elif any(keyword in name_lower for keyword in ["count", "num"]):
    return "integer"
# ... more guessing
```

### Desired Behavior
```python
# Proper type lookup from TypeInferenceEngine results
if column_name in type_mappings:
    return type_mappings[column_name].inferred_type

# Robust fallback without guessing
return UnknownType(
    confidence="unknown",
    context="Could not determine type from static analysis"
)
```

### Testing Considerations
- Ensure all existing tests pass after the refactor
- Add tests for the new UnknownType handling
- Test behavior when TypeInferenceEngine finds vs. doesn't find type information
- Verify that warnings are properly generated when types are unknown
