.PHONY: test test-verbose lint format check install clean build help

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
test:
	uv run --extra dev pytest

test-verbose:
	uv run --extra dev pytest -v

# Code quality
lint:
	uv run --extra dev ruff check --fix

format:
	uv run --extra dev ruff format

check: lint format test
	@echo "All checks passed!"

# Dependencies
install:
	uv sync --extra dev

# Cleanup
clean:
	rm -rf dist/
	rm -rf build/
	rm -rf *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Build
build:
	uv run python -m build
