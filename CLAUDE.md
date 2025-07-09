# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a PySpark Transform Registry package that provides MLflow integration for logging, versioning, and retrieving PySpark data transformation functions. The package allows developers to persist transform functions with metadata and reload them later for reproducible data processing pipelines.

### Project Goals
**Primary Goal**: Create a PySpark Transform Registry that integrates with MLflow to provide versioning, persistence, and retrieval of PySpark data transformation functions.

**Key Objectives**:
1. **Function Persistence**: Log PySpark transform functions as artifacts in MLflow with complete source code and metadata
2. **Reproducible Pipelines**: Enable reloading of previously logged transform functions for consistent data processing workflows
3. **Version Management**: Track multiple versions of transform functions with search and filtering capabilities
4. **Type Safety**: Validate transform inputs using Python type hints to ensure data compatibility
5. **Metadata Preservation**: Capture and preserve function signatures, parameter information, return types, and docstrings
6. **MLflow Integration**: Leverage MLflow's experiment tracking, artifact storage, and search capabilities for transform management

**Technical Approach**: Uses Python's `inspect` module for function introspection, wraps logged functions with imports and metadata headers for standalone execution, implements round-trip testing to ensure function fidelity after persistence/retrieval, and provides validation utilities to check input types against function signatures.

## Core Architecture

The package is structured around four main modules:

- **`core.py`**: Main functionality for logging (`log_transform_function`), loading (`load_transform_function`), and searching (`find_transform_versions`) transform functions using MLflow
- **`metadata.py`**: Utilities for extracting function metadata, type annotations, and generating well-formatted source code with documentation headers
- **`validation.py`**: Input validation utilities using type hints to ensure transform functions receive appropriate input types
- **`__init__.py`**: Package exports and public API

## Development Commands

### Testing
```bash
# Run all tests (using Makefile)
make test

# Run tests with verbose output
make test-verbose

# Run specific test file
uv run --extra dev pytest tests/test_transform_registry.py

# Run specific test function
uv run --extra dev pytest tests/test_transform_registry.py::test_mlflow_round_trip_execution
```

### Code Quality
```bash
# Run all code quality checks (lint, format, test)
make check

# Run linting with auto-fix
make lint

# Run code formatting
make format

# Install pre-commit hooks (run once after cloning)
uv run --extra dev pre-commit install

# Run pre-commit hooks on all files manually
uv run --extra dev pre-commit run --all-files
```

### Package Management
```bash
# Install dependencies with dev extras
make install

# Or manually with uv
uv sync --extra dev

# Build package
make build
```

## Testing Environment Requirements

- **Java 17+** with security manager enabled (configured in test fixtures)
- **PySpark** with local[2] execution mode for testing
- **MLflow** with temporary local tracking for test isolation
- Tests use session-scoped Spark fixtures for efficiency with function-scoped cleanup

## Key Implementation Details

### MLflow Integration Pattern
- Functions are logged as text artifacts with metadata headers
- Metadata includes parameter info, return types, and docstrings
- Search functionality uses MLflow's parameter and tag filtering
- Round-trip execution testing ensures function fidelity

### Type System Integration
- Uses `typing.get_type_hints()` for runtime type information
- Resolves fully qualified type names for accurate validation
- Supports PySpark DataFrame type validation through `pydoc.locate()`

### Code Generation
- Automatically wraps logged functions with import statements and metadata comments
- Preserves original function signatures and behavior
- Uses `textwrap.dedent()` for consistent indentation handling

## Git Workflow

This repository follows a branch-based development workflow with automated code quality checks:

1. **Start with clean main**: Ensure tests pass on main before creating branch
2. **Setup pre-commit hooks**: Run `uv run --extra dev pre-commit install` (once per clone)
3. **Create feature branch**: Work on dedicated branch for each feature/fix
4. **Commit frequently**: Make small, focused commits for easy rollback
5. **Add tests**: Write tests for new features and discovered bugs
6. **Verify tests pass**: Run full test suite before merging
7. **Squash merge**: Use squash commits when merging to main for clean history

**Note**: Pre-commit hooks automatically run linting, formatting, and other checks on every commit. If they fail, the commit is rejected until issues are fixed.

### Frequent Commits for Claude Code

When working with Claude Code, commit frequently to preserve progress:

- **Commit after completing logical units of work** (e.g., after implementing a function, fixing tests, etc.)
- **Commit when tests are passing** - this ensures we don't lose working code
- **Don't ask permission to commit** - if tests pass and we're on a branch, commit automatically
- **Use descriptive commit messages** that explain the change and its purpose
- **Include the Claude Code footer** in commit messages for tracking

### Preserving Commit Messages When Fixing Linting/Formatting

When pre-commit hooks or linting tools modify code during a commit:

- **Always preserve the original commit message** - don't change the message to focus on linting fixes
- **If linting changes are substantial**, mention them briefly in the commit body, not the title
- **Example**: If implementing a new feature triggers formatting changes, keep the commit message about the feature, not the formatting

**Good**:
```
Add automatic transform type detection and API improvements

- Implement _detect_transform_type function for DataFrame/Column/Custom detection
- Add keyword-only argument enforcement to log_transform_function
- Auto-detect function names when not provided
- Update tests for new API features

🤖 Generated with [Claude Code](https://claude.ai/code)
```

**Bad**:
```
Fix linting errors and formatting issues

🤖 Generated with [Claude Code](https://claude.ai/code)
```

### Workflow Commands
```bash
# One-time setup after cloning
make install
uv run --extra dev pre-commit install

# Verify tests pass before starting
make test

# Create and switch to feature branch
git checkout -b feature/your-feature-name

# Work and commit frequently (pre-commit hooks run automatically)
git add .
git commit -m "descriptive commit message"

# Before merging, ensure all checks pass
make check

# Merge with squash commit
git checkout main
git merge --squash feature/your-feature-name
git commit -m "Add feature: description"
```

## Language Server Integration

### When to Use Language Server Tools

**After writing or modifying Python code**, use language server tools to optimize code quality and catch errors:

1. **Check for errors and warnings**: Use `mcp__language-server__diagnostics` to catch:
   - Syntax errors and type mismatches
   - Undefined variables and import issues
   - Unused imports/variables
   - Missing type annotations

2. **Validate function signatures**: Use `mcp__language-server__hover` to verify:
   - Correct parameter types and return annotations
   - Function documentation and docstrings
   - Type information at specific code positions

3. **Find all usages before refactoring**: Use `mcp__language-server__references` to:
   - Identify all places a function/class is used
   - Ensure safe refactoring without breaking dependencies
   - Understand code impact before changes

4. **Safe symbol renaming**: Use `mcp__language-server__rename_symbol` for:
   - Renaming functions, classes, variables across the entire codebase
   - Maintaining consistency without manual find/replace errors

### Recommended Workflow

```bash
# After writing/editing Python code:
1. Run diagnostics on modified files to catch errors
2. Fix any warnings or errors found
3. Use hover info to verify type annotations
4. Before major refactoring, check references
5. Use rename_symbol for safe renaming operations
```

### Example Usage Patterns

- **Post-implementation**: Always run diagnostics after implementing new functions
- **Before commits**: Check diagnostics on all modified files
- **During refactoring**: Use references to understand impact, rename_symbol for changes
- **Type validation**: Use hover to verify complex type annotations are correct

## Test Coverage Areas

- Metadata extraction and type resolution
- MLflow logging and artifact management
- Function loading and dynamic execution
- Input validation with type checking
- End-to-end round-trip execution verification
