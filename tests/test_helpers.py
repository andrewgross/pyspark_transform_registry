import tempfile
from pathlib import Path


from pyspark_transform_registry.helpers import (
    _extract_package_names,
    _filter_requirements,
    _normalize_package_name,
)


class TestPackageNameNormalization:
    """Test package name normalization function."""

    def test_normalize_basic_package_name(self):
        """Test normalizing a basic package name."""
        assert _normalize_package_name("numpy") == "numpy"
        assert _normalize_package_name("NUMPY") == "numpy"
        assert _normalize_package_name("NumPy") == "numpy"

    def test_normalize_package_name_with_underscores(self):
        """Test normalizing package names with underscores."""
        assert _normalize_package_name("my_package") == "my-package"
        assert (
            _normalize_package_name("some_long_package_name")
            == "some-long-package-name"
        )
        assert _normalize_package_name("test_package_123") == "test-package-123"

    def test_normalize_package_name_with_dashes(self):
        """Test normalizing package names with dashes (should remain unchanged)."""
        assert _normalize_package_name("my-package") == "my-package"
        assert (
            _normalize_package_name("some-long-package-name")
            == "some-long-package-name"
        )

    def test_normalize_package_name_with_whitespace(self):
        """Test normalizing package names with whitespace."""
        assert _normalize_package_name("  numpy  ") == "numpy"
        assert _normalize_package_name("\tmy_package\n") == "my-package"

    def test_normalize_mixed_case_and_separators(self):
        """Test normalizing package names with mixed case and separators."""
        assert _normalize_package_name("My_Package-Name") == "my-package-name"
        assert _normalize_package_name("TEST_package-123") == "test-package-123"


class TestExtractPackageNames:
    """Test package name extraction from requirements files."""

    def test_extract_simple_packages(self):
        """Test extracting simple package names."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("numpy==1.21.0\n")
            f.write("pandas>=1.3.0\n")
            f.write("requests~=2.25.0\n")
            f.flush()

            packages = _extract_package_names(f.name)
            assert packages == [
                ("numpy", "numpy==1.21.0"),
                ("pandas", "pandas>=1.3.0"),
                ("requests", "requests~=2.25.0"),
            ]

        Path(f.name).unlink()

    def test_extract_packages_with_extras(self):
        """Test extracting package names with extras."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("requests[security]==2.25.0\n")
            f.write("pandas[all]>=1.3.0\n")
            f.write("sqlalchemy[postgresql,mysql]\n")
            f.flush()

            packages = _extract_package_names(f.name)
            assert packages == [
                ("requests", "requests[security]==2.25.0"),
                ("pandas", "pandas[all]>=1.3.0"),
                ("sqlalchemy", "sqlalchemy[postgresql,mysql]"),
            ]

        Path(f.name).unlink()

    def test_extract_packages_with_environment_markers(self):
        """Test extracting package names with environment markers."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write('numpy==1.21.0; python_version >= "3.7"\n')
            f.write("pandas>=1.3.0; sys_platform == 'win32'\n")
            f.write('requests~=2.25.0; extra == "dev"\n')
            f.flush()

            packages = _extract_package_names(f.name)
            assert packages == [
                ("numpy", 'numpy==1.21.0; python_version >= "3.7"'),
                ("pandas", "pandas>=1.3.0; sys_platform == 'win32'"),
                ("requests", 'requests~=2.25.0; extra == "dev"'),
            ]

        Path(f.name).unlink()

    def test_extract_editable_packages_with_egg(self):
        """Test extracting editable packages with #egg= fragment."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("-e git+https://github.com/user/repo.git#egg=my_package\n")
            f.write(
                "--editable=git+https://github.com/user/other.git#egg=other-package\n"
            )
            f.write("-e file:///path/to/local#egg=local_pkg\n")
            f.flush()

            packages = _extract_package_names(f.name)
            assert packages == [
                (
                    "my-package",
                    "-e git+https://github.com/user/repo.git#egg=my_package",
                ),
                (
                    "other-package",
                    "--editable=git+https://github.com/user/other.git#egg=other-package",
                ),
                ("local-pkg", "-e file:///path/to/local#egg=local_pkg"),
            ]

        Path(f.name).unlink()

    def test_extract_packages_ignore_comments_and_empty_lines(self):
        """Test that comments and empty lines are ignored."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("# This is a comment\n")
            f.write("\n")
            f.write("numpy==1.21.0\n")
            f.write("# Another comment\n")
            f.write("\n")
            f.write("pandas>=1.3.0\n")
            f.write("   # Indented comment\n")
            f.flush()

            packages = _extract_package_names(f.name)
            assert packages == [("numpy", "numpy==1.21.0"), ("pandas", "pandas>=1.3.0")]

        Path(f.name).unlink()

    def test_extract_packages_complex_requirements(self):
        """Test extracting from a complex requirements file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("# Core dependencies\n")
            f.write("numpy==1.21.0\n")
            f.write("pandas[all]>=1.3.0\n")
            f.write("\n")
            f.write("# Web dependencies\n")
            f.write('requests[security]~=2.25.0; python_version >= "3.7"\n')
            f.write("flask==2.0.0\n")
            f.write("\n")
            f.write("# Development dependencies\n")
            f.write("-e git+https://github.com/user/dev-tool.git#egg=dev_tool\n")
            f.write("pytest>=6.0.0\n")
            f.flush()

            packages = _extract_package_names(f.name)
            assert packages == [
                ("numpy", "numpy==1.21.0"),
                ("pandas", "pandas[all]>=1.3.0"),
                ("requests", 'requests[security]~=2.25.0; python_version >= "3.7"'),
                ("flask", "flask==2.0.0"),
                (
                    "dev-tool",
                    "-e git+https://github.com/user/dev-tool.git#egg=dev_tool",
                ),
                ("pytest", "pytest>=6.0.0"),
            ]

        Path(f.name).unlink()

    def test_extract_packages_with_underscores_and_dashes(self):
        """Test that package names with underscores are normalized to dashes."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("my_package==1.0.0\n")
            f.write("another-package>=2.0.0\n")
            f.write("Mixed_Case-Package\n")
            f.flush()

            packages = _extract_package_names(f.name)
            assert packages == [
                ("my-package", "my_package==1.0.0"),
                ("another-package", "another-package>=2.0.0"),
                ("mixed-case-package", "Mixed_Case-Package"),
            ]

        Path(f.name).unlink()


class TestFilterRequirements:
    """Test requirements filtering functionality."""

    def test_filter_simple_requirements(self):
        """Test filtering simple requirements."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("numpy==1.21.0\n")
            f.write("pandas>=1.3.0\n")
            f.write("requests~=2.25.0\n")
            f.flush()

            filtered = _filter_requirements(f.name, ["pandas"])
            assert filtered == ["numpy==1.21.0", "requests~=2.25.0"]

        Path(f.name).unlink()

    def test_filter_requirements_with_normalization(self):
        """Test filtering with package name normalization."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("my_package==1.0.0\n")
            f.write("another-package>=2.0.0\n")
            f.write("third_pkg\n")
            f.flush()

            # Test excluding "my-package" should match "my_package" in file due to normalization
            filtered = _filter_requirements(f.name, ["my-package"])
            assert filtered == ["another-package>=2.0.0", "third_pkg"]

            # Test excluding "another_package" should match "another-package" in file due to normalization
            filtered = _filter_requirements(f.name, ["another_package"])
            assert filtered == ["my_package==1.0.0", "third_pkg"]

        Path(f.name).unlink()

    def test_filter_multiple_packages(self):
        """Test filtering multiple packages."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("numpy==1.21.0\n")
            f.write("pandas>=1.3.0\n")
            f.write("requests~=2.25.0\n")
            f.write("flask==2.0.0\n")
            f.write("pytest>=6.0.0\n")
            f.flush()

            filtered = _filter_requirements(f.name, ["pandas", "pytest"])
            assert filtered == ["numpy==1.21.0", "requests~=2.25.0", "flask==2.0.0"]

        Path(f.name).unlink()

    def test_filter_no_matches(self):
        """Test filtering when no packages match exclusion list."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("numpy==1.21.0\n")
            f.write("pandas>=1.3.0\n")
            f.write("requests~=2.25.0\n")
            f.flush()

            filtered = _filter_requirements(f.name, ["nonexistent", "also-missing"])
            assert filtered == ["numpy==1.21.0", "pandas>=1.3.0", "requests~=2.25.0"]

        Path(f.name).unlink()

    def test_filter_all_packages(self):
        """Test filtering when all packages are excluded."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("numpy==1.21.0\n")
            f.write("pandas>=1.3.0\n")
            f.flush()

            filtered = _filter_requirements(f.name, ["numpy", "pandas"])
            assert filtered == []

        Path(f.name).unlink()

    def test_filter_with_complex_requirements(self):
        """Test filtering with complex requirements including editable installs."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("# Core dependencies\n")
            f.write("numpy==1.21.0\n")
            f.write("pandas[all]>=1.3.0\n")
            f.write("-e git+https://github.com/user/dev-tool.git#egg=dev_tool\n")
            f.write('requests[security]~=2.25.0; python_version >= "3.7"\n')
            f.flush()

            filtered = _filter_requirements(f.name, ["dev-tool", "pandas"])
            assert len(filtered) == 2
            assert "numpy==1.21.0" in filtered
            assert 'requests[security]~=2.25.0; python_version >= "3.7"' in filtered

        Path(f.name).unlink()
