.PHONY: test test-verbose lint format check install clean build help setup publish

# Default target
help:
	@echo "Available commands:"
	@echo "  test         - Run all tests"
	@echo "  test-verbose - Run all tests with verbose output"
	@echo "  lint         - Run linting checks"
	@echo "  format       - Format code"
	@echo "  check        - Run all code quality checks"
	@echo "  install      - Install dependencies"
	@echo "  clean        - Clean build artifacts"
	@echo "  build        - Build package"

# Testing
pre-commit:
	uv run pre-commit run --all-files

test:
	uv run --dev pytest tests/

test-verbose:
	uv run --dev pytest -v tests/

test-debug:
	uv run --dev pytest -v tests/ --pdb

# Code quality
lint:
	uv run --dev ruff check --fix

format:
	uv run --dev ruff format

check: lint format test
	@echo "All checks passed!"

setup: install

install:
	uv sync --dev
	uv run pre-commit install

build: clean
	@echo "Building package..."
	uv build
	@echo "Build complete!"

publish: build
	@echo "Publishing package..."
	uv publish
	@echo "Publish complete!"


clean:
	@echo "Cleaning up..."
	@rm -rf __pycache__/ .pytest_cache/
	@rm -rf category_indexer/
	@rm -rf dist/ build/
	@find . -name "*.pyc" -delete
	@find . -name "*.pkl" -delete
	@find . -name "*.egg-info" -type d -exec rm -rf {} + 2>/dev/null || true
	@find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	@echo "Done!"
