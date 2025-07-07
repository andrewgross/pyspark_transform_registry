# GitHub Actions Configuration

This directory contains GitHub Actions workflows and configurations for automated CI/CD processes.

## Workflows

### 🧪 Tests (`tests.yml`)
- **Triggers**: Push to main, Pull requests to main
- **Purpose**: Runs the test suite using pytest
- **Features**:
  - Tests on Python 3.11 and 3.12
  - Sets up Java 17 (required for PySpark)
  - Uses uv for dependency management
  - Generates coverage reports
  - Uploads coverage to Codecov

### 🔍 Code Quality (`code-quality.yml`)
- **Triggers**: Push to main, Pull requests to main
- **Purpose**: Runs code quality checks using pre-commit hooks
- **Features**:
  - Runs ruff for linting and formatting
  - Applies pyupgrade for modern Python syntax
  - Checks trailing whitespace, YAML syntax, etc.
  - Caches pre-commit environments for faster runs

### 📦 Build (`build.yml`)
- **Triggers**: Push to main, Pull requests to main, Releases
- **Purpose**: Validates package building and installation
- **Features**:
  - Builds wheel and source distributions
  - Validates package metadata with twine
  - Tests clean installation in isolated environment
  - Uploads build artifacts for inspection

### 🔒 Security (`security.yml`)
- **Triggers**: Push to main, Pull requests to main, Weekly schedule
- **Purpose**: Scans for security vulnerabilities
- **Features**:
  - Uses Safety to check for known vulnerabilities in dependencies
  - Runs Bandit for Python security linting
  - Performs CodeQL analysis for security issues
  - Generates security reports

## Configuration Files

### 📋 Dependabot (`dependabot.yml`)
- **Purpose**: Automatically updates dependencies
- **Features**:
  - Weekly updates for Python dependencies
  - Weekly updates for GitHub Actions
  - Assigns updates to @maintainers team
  - Limits open PRs to prevent spam

## Getting Started

### For Developers

1. **Local Setup**: Install pre-commit hooks locally:
   ```bash
   uv run pre-commit install
   ```

2. **Running Tests Locally**:
   ```bash
   uv sync --dev
   uv run pytest tests/ -v
   ```

3. **Code Quality Checks**:
   ```bash
   uv run pre-commit run --all-files
   ```

### For Maintainers

1. **Required Status Checks**: Consider requiring the following checks to pass before merging:
   - Tests (both Python 3.11 and 3.12)
   - Code Quality
   - Build validation

2. **Security**: Review security workflow outputs regularly, especially:
   - Safety reports for dependency vulnerabilities
   - CodeQL alerts for potential security issues

3. **Dependabot**: Review and merge dependency updates regularly to maintain security

## Customization

### Adding New Checks
To add new quality checks, update `.pre-commit-config.yaml` and they will automatically run in the Code Quality workflow.

### Modifying Test Configuration
Update `tests/conftest.py` for test setup changes, or `pyproject.toml` for dependency changes.

### Branch Protection
Consider setting up branch protection rules requiring:
- Status checks to pass
- Up-to-date branches
- Require review from code owners

## Caching

All workflows use caching to improve performance:
- Python dependencies (uv cache)
- Pre-commit environments
- Build artifacts

Cache keys are based on lock files and configuration, ensuring cache invalidation when dependencies change.