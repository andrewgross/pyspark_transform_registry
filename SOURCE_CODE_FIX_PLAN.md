# Source Code Visibility Fix Plan

## Problem Summary
You have a **source code visibility issue** in the PySpark Transform Registry when loading transforms that were defined from inline Python functions (vs file-based functions). The `get_source()` method in `core.py` calls `inspect.getsource(original_func)` which fails for certain types of function definitions.

## Root Cause Analysis Complete
**Current Implementation**:
- `PySparkTransformModel.__init__()` stores `self.function_source = inspect.getsource(transform_func)`
- `load_function()` creates a `get_source()` method that calls `inspect.getsource(original_func)` on the loaded function
- This fails for dynamically created functions, functions defined in REPL/notebooks, or certain inline definitions

**Key Files Involved**:
- `src/pyspark_transform_registry/core.py:284-293` - `get_source()` implementation
- `src/pyspark_transform_registry/model_wrapper.py:35` - Source storage during registration
- No existing tests for source code retrieval functionality

## Implementation Plan

**Phase 1: Store Source Code as MLflow Tag/Artifact**
1. Modify `PySparkTransformModel.__init__()` in `model_wrapper.py` to capture source code during registration
2. Update `register_function()` in `core.py` to store source code as MLflow tag: `"function_source"`
3. Modify `load_function()` to retrieve source from MLflow tags instead of re-inspecting
4. Add fallback logic: try stored source first, then `inspect.getsource()` as backup

**Phase 2: Add Comprehensive Test Coverage**
1. Create `tests/test_source_code_inspection.py` with tests for success/failure scenarios
2. Add test fixtures for functions that can't be inspected
3. Test error handling and graceful degradation

**Phase 3: Alternative Solution (if needed)**
If tag storage has size limits, implement temporary file approach with MLflow artifacts.

## Next Session Instruction
"I need to debug and fix the source code visibility issue in the PySpark Transform Registry. The `get_source()` method fails for inline function definitions. Implement the plan to store function source code as MLflow tags during registration and retrieve it during loading, with fallback to `inspect.getsource()`. Start with modifying `model_wrapper.py` and `core.py`, then add comprehensive tests."

## Technical Details

### Current Flow
1. Registration: `PySparkTransformModel.__init__()` calls `inspect.getsource(transform_func)` and stores in `self.function_source`
2. Loading: `load_function()` creates `get_source()` that calls `inspect.getsource(original_func)` again
3. **Problem**: Step 2 fails for inline/dynamic functions even though step 1 succeeded

### Proposed Fix
1. Registration: Store captured source code as MLflow tag `"function_source"`
2. Loading: Retrieve source from MLflow tags first, fallback to `inspect.getsource()` if not found
3. Testing: Add comprehensive tests for both success and failure scenarios

### Files to Modify
- `src/pyspark_transform_registry/model_wrapper.py` - Capture source during init
- `src/pyspark_transform_registry/core.py` - Store as tag, retrieve in get_source()
- `tests/test_source_code_inspection.py` - New test file
