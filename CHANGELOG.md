# Changelog

All notable changes to this project will be documented in this file.

## [0.6.0] - 2025-09-05

### Features
- Add support for loading functions via URIs with `load_function_from_uri()` and `load_transform_from_uri()` (11891a3)
- Add `register_transform()` and `load_transform()` convenience aliases for better API consistency (11891a3)
- Add `transform_uri` attribute to registered models for clearer URI handling with transforms:/ prefix (11891a3)
- Add debug testing target in Makefile with `make test-debug` (11891a3)

### Enhancements
- Improve model URI handling logic with support for both models:/ and transforms:/ prefixes (11891a3)
- Enhanced `get_latest_function_version()` with better error handling for missing models (11891a3)
- Default `validate_input` to `False` for better usability in `load_function()` (11891a3)
- Return version as string type from `get_latest_function_version()` for consistency (11891a3)

### Breaking Changes
- Completely removed static analysis and type inference functionality to simplify the core library (c8c934c)
- Removed complex schema constraint validation system (c8c934c)
- Removed runtime validation engine (c8c934c)
- Eliminated examples directory and comprehensive documentation (c8c934c)
- Removed metadata utilities and advanced introspection features (c8c934c)
- Stripped down to basic function registration and loading only (c8c934c)

### Documentation
- Removed extensive examples and documentation as part of simplification (c8c934c)

## [0.5.0] - 2025-08-19

### Breaking Changes
- Relaxed Python version requirement from >=3.11 to >=3.9 for broader compatibility (dcf3bc2)

## [0.4.0] - 2025-07-24

### Features
- Add dummy MLflow signature support for improved model registry integration (7424032)

## [0.3.0] - 2025-07-23

### Features
- Add `get_latest_function_version()` function to retrieve the most recent version of registered models (ff2d92b)

### Dependencies
- Move PySpark from main dependencies to dev requirements to simplify Databricks installation (a91bc07)
- Users can now install the package without PySpark for Databricks environments where PySpark is pre-installed (a91bc07)

## [0.2.0] - 2025-07-23

### Dependencies
- Move PySpark to development dependencies to support Databricks installations (a91bc07)

## [0.1.0] - 2025-07-22

### Features
- Initial release of PySpark Transform Registry with MLflow integration
- Function registration with `register_function()` for persisting transform functions (92c2eaa)
- Function loading with `load_function()` for retrieving registered transforms (92c2eaa)
- Support for both direct function and file-based registration patterns (92c2eaa)
- MLflow model registry integration for version management (92c2eaa)
- Static analysis framework using LibCST for schema inference (92c2eaa)
- Runtime validation engine for DataFrame schema constraints (92c2eaa)
- Comprehensive metadata preservation including function signatures and documentation (92c2eaa)
- Type safety validation using inferred schema constraints (92c2eaa)

### Dependencies
- MLflow >=3.0.0 for model registry and artifact storage
- PySpark <4.0 for DataFrame processing
- Python >=3.11 requirement
