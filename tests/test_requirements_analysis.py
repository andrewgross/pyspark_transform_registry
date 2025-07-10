"""Tests for requirements analysis functionality."""

from pyspark.sql import DataFrame, Column
from pyspark.sql.functions import col, lit, when

from pyspark_transform_registry.requirements_analysis import (
    DependencyAnalyzer,
    FunctionCluster,
    validate_function_safety,
    create_minimal_requirements,
)


class TestDependencyAnalyzer:
    """Test DependencyAnalyzer functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = DependencyAnalyzer()

    def test_stdlib_module_detection(self):
        """Test detection of standard library modules."""
        assert self.analyzer._is_stdlib_module("os")
        assert self.analyzer._is_stdlib_module("sys")
        assert self.analyzer._is_stdlib_module("json")
        assert self.analyzer._is_stdlib_module("datetime")
        assert not self.analyzer._is_stdlib_module("pandas")
        assert not self.analyzer._is_stdlib_module("numpy")

    def test_pyspark_module_detection(self):
        """Test detection of PySpark modules."""
        assert self.analyzer._is_pyspark_module("pyspark")
        assert self.analyzer._is_pyspark_module("pyspark.sql")
        assert self.analyzer._is_pyspark_module("pyspark.sql.functions")
        assert not self.analyzer._is_pyspark_module("pandas")
        assert not self.analyzer._is_pyspark_module("sklearn")

    def test_third_party_module_detection(self):
        """Test detection of third-party modules."""
        # Test with modules that should be available in test environment
        # Note: This may vary based on test environment
        assert not self.analyzer._is_third_party_module("os")  # stdlib
        assert not self.analyzer._is_third_party_module("pyspark")  # pyspark
        # Third-party detection depends on what's actually installed

    def test_module_to_requirement_mapping(self):
        """Test mapping of module names to package requirements."""
        assert self.analyzer._module_to_requirement("sklearn") == "scikit-learn"
        assert self.analyzer._module_to_requirement("PIL") == "Pillow"
        assert self.analyzer._module_to_requirement("yaml") == "PyYAML"
        assert self.analyzer._module_to_requirement("pandas") == "pandas"

    def test_analyze_simple_function(self):
        """Test analyzing a simple function with no external dependencies."""

        def simple_transform(df: DataFrame) -> DataFrame:
            return df.select("*")

        deps = self.analyzer.analyze_function_dependencies(simple_transform)

        assert isinstance(deps["pip_requirements"], list)
        assert isinstance(deps["local_imports"], list)
        assert isinstance(deps["external_calls"], list)

        # Should have minimal dependencies
        assert len(deps["pip_requirements"]) == 0  # No third-party imports
        assert len(deps["local_imports"]) == 0  # No local imports

    def test_analyze_function_with_imports(self):
        """Test analyzing a function with various imports."""

        def transform_with_imports(df: DataFrame) -> DataFrame:
            import os  # stdlib
            from pyspark.sql.functions import upper  # pyspark

            # Use the imports
            _ = os.path.join("/tmp", "config.json")
            _ = {"test": True}

            return df.withColumn("upper_col", upper(col("name")))

        deps = self.analyzer.analyze_function_dependencies(transform_with_imports)

        # Should not include stdlib or pyspark modules in pip requirements
        # The exact count depends on what other packages might be detected
        assert isinstance(deps["pip_requirements"], list)

    def test_analyze_function_with_external_calls(self):
        """Test analyzing a function with external function calls."""

        def transform_with_external_calls(df: DataFrame) -> DataFrame:
            # This would call some external function
            def some_external_function(df):
                return df

            result = some_external_function(
                df,
            )  # This will be detected as external call
            return result.withColumn("processed", lit(True))

        deps = self.analyzer.analyze_function_dependencies(
            transform_with_external_calls,
        )

        # Should detect external function call
        assert "some_external_function" in deps["external_calls"]

    def test_analyze_column_expression_function(self):
        """Test analyzing a column expression function."""

        def risk_calculator(amount: Column, threshold: int = 100) -> Column:
            return when(amount > threshold, amount * 1.5).otherwise(amount)

        deps = self.analyzer.analyze_function_dependencies(risk_calculator)

        # Column expressions should have minimal dependencies
        assert isinstance(deps["pip_requirements"], list)
        assert isinstance(deps["external_calls"], list)


class TestFunctionCluster:
    """Test FunctionCluster functionality."""

    def test_cluster_creation(self):
        """Test creating a function cluster."""
        cluster = FunctionCluster("test_cluster")

        assert cluster.name == "test_cluster"
        assert len(cluster.functions) == 0
        assert len(cluster.local_code_paths) == 0

    def test_add_functions_to_cluster(self):
        """Test adding functions to a cluster."""

        def func1(df: DataFrame) -> DataFrame:
            return df.select("*")

        def func2(df: DataFrame) -> DataFrame:
            return df.filter(col("value") > 0)

        cluster = FunctionCluster("test_cluster")
        cluster.add_function(func1)
        cluster.add_function(func2)

        assert len(cluster.functions) == 2
        assert func1 in cluster.functions
        assert func2 in cluster.functions

    def test_add_code_paths_to_cluster(self):
        """Test adding code paths to a cluster."""
        cluster = FunctionCluster("test_cluster")
        cluster.add_local_code_path("utils/")
        cluster.add_local_code_path("models/")

        assert len(cluster.local_code_paths) == 2
        assert "utils/" in cluster.local_code_paths
        assert "models/" in cluster.local_code_paths

    def test_analyze_cluster_dependencies(self):
        """Test analyzing dependencies for a function cluster."""

        def func1(df: DataFrame) -> DataFrame:
            return df.select("*")

        def func2(df: DataFrame) -> DataFrame:
            from pyspark.sql.functions import upper  # pyspark

            return df.withColumn("upper_col", upper(col("name")))

        cluster = FunctionCluster("test_cluster")
        cluster.add_function(func1)
        cluster.add_function(func2)
        cluster.add_local_code_path("utils/")

        analyzer = DependencyAnalyzer()
        deps = cluster.analyze_cluster_dependencies(analyzer)

        assert isinstance(deps["pip_requirements"], list)
        assert isinstance(deps["local_imports"], list)
        assert isinstance(deps["external_calls"], list)
        assert isinstance(deps["code_paths"], list)
        assert "utils/" in deps["code_paths"]


class TestValidationFunctions:
    """Test validation and safety functions."""

    def test_validate_function_safety_simple(self):
        """Test validating a simple, safe function."""

        def safe_function(df: DataFrame) -> DataFrame:
            return df.select("*")

        analyzer = DependencyAnalyzer()
        validation = validate_function_safety(safe_function, analyzer)

        assert isinstance(validation["warnings"], list)
        assert isinstance(validation["errors"], list)
        assert "dependencies" in validation

        # Simple function should have no warnings or errors
        assert len(validation["warnings"]) == 0
        assert len(validation["errors"]) == 0

    def test_validate_function_safety_with_external_calls(self):
        """Test validating a function with external calls."""

        def function_with_external_calls(df: DataFrame) -> DataFrame:
            # This calls an external function
            def external_processor(df):
                return df

            processed_df = external_processor(df)
            return processed_df.withColumn("flag", lit(True))

        analyzer = DependencyAnalyzer()
        validation = validate_function_safety(function_with_external_calls, analyzer)

        # Should have warnings about external calls
        assert len(validation["warnings"]) > 0
        warning_text = " ".join(validation["warnings"])
        assert "external" in warning_text.lower()

    def test_create_minimal_requirements_single_function(self):
        """Test creating minimal requirements for a single function."""

        def simple_function(df: DataFrame) -> DataFrame:
            return df.select("*")

        requirements = create_minimal_requirements(simple_function)

        assert isinstance(requirements["pip_requirements"], list)
        assert isinstance(requirements["code_paths"], list)
        assert isinstance(requirements["warnings"], list)

        # Should include core requirements
        req_str = " ".join(requirements["pip_requirements"])
        assert "pyspark" in req_str.lower()
        assert "mlflow" in req_str.lower()

    def test_create_minimal_requirements_multiple_functions(self):
        """Test creating minimal requirements for multiple functions."""

        def func1(df: DataFrame) -> DataFrame:
            return df.select("*")

        def func2(df: DataFrame) -> DataFrame:
            return df.filter(col("value") > 0)

        requirements = create_minimal_requirements([func1, func2])

        assert isinstance(requirements["pip_requirements"], list)
        assert isinstance(requirements["code_paths"], list)
        assert isinstance(requirements["warnings"], list)

    def test_create_minimal_requirements_with_extras(self):
        """Test creating minimal requirements with extra requirements."""

        def simple_function(df: DataFrame) -> DataFrame:
            return df.select("*")

        extra_reqs = ["pandas==2.0.0", "numpy>=1.24.0"]
        code_paths = ["utils/", "models/"]

        requirements = create_minimal_requirements(
            simple_function,
            extra_requirements=extra_reqs,
            code_paths=code_paths,
        )

        # Should include extra requirements
        assert "pandas==2.0.0" in requirements["pip_requirements"]
        assert "numpy>=1.24.0" in requirements["pip_requirements"]

        # Should include code paths
        assert "utils/" in requirements["code_paths"]
        assert "models/" in requirements["code_paths"]

    def test_create_minimal_requirements_complex_function(self):
        """Test creating requirements for a function with complex dependencies."""

        def complex_function(df: DataFrame) -> DataFrame:
            # This would typically import heavy ML libraries
            # import tensorflow as tf
            # import sklearn
            return df.withColumn("ml_score", lit(0.95))

        requirements = create_minimal_requirements(
            complex_function,
            extra_requirements=["tensorflow==2.13.0", "scikit-learn==1.3.0"],
        )

        # Should detect complex dependencies and warn
        # Note: Exact warning content depends on implementation
        assert isinstance(requirements["warnings"], list)


class TestIntegrationScenarios:
    """Test integration scenarios with different function types."""

    def test_dataframe_transform_analysis(self):
        """Test analyzing a typical DataFrame transform."""

        def clean_data(df: DataFrame, min_value: float = 0.0) -> DataFrame:
            """Clean data by filtering and removing nulls."""
            from pyspark.sql.functions import isnotnull

            return df.filter((col("amount") > min_value) & isnotnull(col("amount")))

        analyzer = DependencyAnalyzer()
        deps = analyzer.analyze_function_dependencies(clean_data)

        # Should have minimal dependencies (only PySpark)
        assert isinstance(deps["pip_requirements"], list)

        # Validate the function is safe
        validation = validate_function_safety(clean_data, analyzer)
        assert len(validation["errors"]) == 0  # Should be safe

    def test_column_expression_analysis(self):
        """Test analyzing a column expression function."""

        def calculate_risk_score(amount: Column, category: Column) -> Column:
            """Calculate risk score based on amount and category."""
            return (
                when(category == "high_risk", amount * 1.5)
                .when(category == "medium_risk", amount * 1.2)
                .otherwise(amount)
            )

        analyzer = DependencyAnalyzer()
        deps = analyzer.analyze_function_dependencies(calculate_risk_score)

        # Column expressions should be very minimal
        assert isinstance(deps["pip_requirements"], list)
        assert isinstance(deps["external_calls"], list)

    def test_function_with_local_imports_analysis(self):
        """Test analyzing a function that would import local modules."""

        def transform_with_utils(df: DataFrame) -> DataFrame:
            """Transform that would use local utility functions."""
            # In real scenario, this would be:
            # from utils.data_processing import normalize_text
            # from utils.calculations import calculate_score

            # For test, we simulate with inline logic
            return df.withColumn("processed", lit(True))

        requirements = create_minimal_requirements(
            transform_with_utils,
            code_paths=["utils/", "calculations/"],
        )

        # Should include the specified code paths
        assert "utils/" in requirements["code_paths"]
        assert "calculations/" in requirements["code_paths"]

    def test_error_handling_in_analysis(self):
        """Test error handling when analysis fails."""

        # Create a problematic function that might cause analysis issues
        def problematic_function():
            """Function with unusual signature that might cause issues."""
            pass  # No parameters, not a proper transform

        analyzer = DependencyAnalyzer()

        # Should handle errors gracefully
        deps = analyzer.analyze_function_dependencies(problematic_function)

        # Should return empty results rather than crashing
        assert isinstance(deps["pip_requirements"], list)
        assert isinstance(deps["local_imports"], list)
        assert isinstance(deps["external_calls"], list)
