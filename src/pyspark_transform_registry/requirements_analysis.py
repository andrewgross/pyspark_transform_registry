"""
Requirements analysis and minimal dependency detection for PySpark transforms.

This module provides utilities to analyze Python functions and extract minimal
sets of dependencies, detect external function calls, and handle function clusters.
"""

import ast
import inspect
import sys
from typing import Optional, Callable, Union
import importlib.util


class DependencyAnalyzer:
    """Analyzes function dependencies and detects minimal requirements."""

    def __init__(self):
        self.stdlib_modules = self._get_stdlib_modules()
        self.pyspark_modules = self._get_pyspark_modules()

    def _get_stdlib_modules(self) -> set[str]:
        """Get set of Python standard library modules."""
        # Core stdlib modules that are always available
        stdlib_modules = {
            "os",
            "sys",
            "collections",
            "itertools",
            "functools",
            "operator",
            "typing",
            "datetime",
            "json",
            "csv",
            "math",
            "random",
            "string",
            "time",
            "uuid",
            "urllib",
            "http",
            "html",
            "xml",
            "email",
            "base64",
            "hashlib",
            "hmac",
            "secrets",
            "subprocess",
            "threading",
            "multiprocessing",
            "concurrent",
            "queue",
            "asyncio",
            "socket",
            "ssl",
            "select",
            "selectors",
            "signal",
            "mmap",
            "ctypes",
            "struct",
            "codecs",
            "locale",
            "gettext",
            "calendar",
            "zoneinfo",
            "configparser",
            "logging",
            "getpass",
            "curses",
            "platform",
            "errno",
            "io",
            "pathlib",
            "glob",
            "tempfile",
            "shutil",
            "stat",
            "fileinput",
            "linecache",
            "shlex",
            "keyword",
            "token",
            "tokenize",
            "ast",
            "symtable",
            "dis",
            "pickletools",
            "formatter",
            "getopt",
            "argparse",
            "copy",
            "pprint",
            "reprlib",
            "enum",
            "types",
            "weakref",
            "gc",
            "inspect",
            "site",
            "user",
            "builtins",
            "__future__",
            "warnings",
            "contextlib",
            "abc",
            "atexit",
            "traceback",
            "tracemalloc",
            "faulthandler",
            "pdb",
            "profile",
            "timeit",
            "trace",
            "py_compile",
            "compileall",
            "dis",
            "pickletools",
            "tabnanny",
            "pydoc",
            "doctest",
            "unittest",
            "test",
        }
        return stdlib_modules

    def _get_pyspark_modules(self) -> set[str]:
        """Get set of PySpark modules."""
        return {
            "pyspark",
            "pyspark.sql",
            "pyspark.sql.functions",
            "pyspark.sql.types",
            "pyspark.sql.window",
            "pyspark.ml",
            "pyspark.mllib",
            "pyspark.streaming",
            "pyspark.context",
            "pyspark.rdd",
            "pyspark.broadcast",
            "pyspark.accumulator",
        }

    def analyze_function_dependencies(self, func: Callable) -> dict[str, list[str]]:
        """
        Analyze a function's dependencies and return minimal requirements.

        Args:
            func: Function to analyze

        Returns:
            Dictionary with 'pip_requirements', 'local_imports', and 'external_calls'
        """
        try:
            source = inspect.getsource(func)
            # Clean up the source code for better parsing
            lines = source.split("\n")
            # Find the first line with the function definition
            def_line_idx = None
            for i, line in enumerate(lines):
                if line.strip().startswith("def "):
                    def_line_idx = i
                    break

            if def_line_idx is not None:
                # Get the indentation of the function definition
                def_line = lines[def_line_idx]
                base_indent = len(def_line) - len(def_line.lstrip())

                # Remove common indentation from all lines
                cleaned_lines = []
                for line in lines[def_line_idx:]:
                    if line.strip():  # Skip empty lines
                        if len(line) >= base_indent:
                            cleaned_lines.append(line[base_indent:])
                        else:
                            cleaned_lines.append(line.lstrip())
                    else:
                        cleaned_lines.append("")

                cleaned_source = "\n".join(cleaned_lines)
            else:
                # Fallback to original source
                cleaned_source = source

            tree = ast.parse(cleaned_source)

            imports = self._extract_imports(tree)
            external_calls = self._extract_external_calls(tree, func)

            # Classify imports
            pip_requirements = []
            local_imports = []

            for module in imports:
                if self._is_third_party_module(module):
                    requirement = self._module_to_requirement(module)
                    if requirement:
                        pip_requirements.append(requirement)
                elif not self._is_stdlib_module(module) and not self._is_pyspark_module(
                    module,
                ):
                    local_imports.append(module)

            return {
                "pip_requirements": list(set(pip_requirements)),
                "local_imports": list(set(local_imports)),
                "external_calls": external_calls,
            }

        except Exception as e:
            print(f"Warning: Could not analyze dependencies for {func.__name__}: {e}")
            return {"pip_requirements": [], "local_imports": [], "external_calls": []}

    def _extract_imports(self, tree: ast.AST) -> set[str]:
        """Extract all import statements from AST."""
        imports = set()

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module.split(".")[0])

        return imports

    def _extract_external_calls(self, tree: ast.AST, func: Callable) -> list[str]:
        """Extract calls to external functions that might not be bundled."""
        external_calls = []

        # Get local function names defined in the same module
        local_functions = set()
        if hasattr(func, "__module__") and func.__module__ in sys.modules:
            module = sys.modules[func.__module__]
            for name in dir(module):
                obj = getattr(module, name)
                if callable(obj) and not name.startswith("_"):
                    local_functions.add(name)

        # Find function calls that might be external
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                call_name = self._get_call_name(node)
                if call_name and call_name not in local_functions:
                    # Check if it's a built-in or imported function
                    if call_name not in dir(__builtins__):
                        external_calls.append(call_name)

        return list(set(external_calls))

    def _get_call_name(self, call_node: ast.Call) -> Optional[str]:
        """Extract function name from call node."""
        if isinstance(call_node.func, ast.Name):
            return call_node.func.id
        elif isinstance(call_node.func, ast.Attribute):
            return call_node.func.attr
        return None

    def _is_stdlib_module(self, module: str) -> bool:
        """Check if module is part of Python standard library."""
        return module in self.stdlib_modules

    def _is_pyspark_module(self, module: str) -> bool:
        """Check if module is part of PySpark."""
        return module in self.pyspark_modules or module.startswith("pyspark")

    def _is_third_party_module(self, module: str) -> bool:
        """Check if module is a third-party package."""
        if self._is_stdlib_module(module) or self._is_pyspark_module(module):
            return False

        # Try to import and check if it's installed
        try:
            importlib.import_module(module)
            return True
        except ImportError:
            return False

    def _module_to_requirement(self, module: str) -> Optional[str]:
        """Convert module name to pip requirement specification."""
        # Map common module names to package names
        module_to_package = {
            "sklearn": "scikit-learn",
            "cv2": "opencv-python",
            "PIL": "Pillow",
            "yaml": "PyYAML",
            "dateutil": "python-dateutil",
        }

        package_name = module_to_package.get(module, module)

        # Try to get version if available
        try:
            spec = importlib.util.find_spec(module)
            if spec and spec.origin:
                # For now, just return package name without version
                # Could be enhanced to detect version from installed package
                return package_name
        except Exception:
            pass

        return package_name


class FunctionCluster:
    """Manages clusters of related functions that should be bundled together."""

    def __init__(self, name: str):
        self.name = name
        self.functions: list[Callable] = []
        self.shared_dependencies: set[str] = set()
        self.local_code_paths: list[str] = []

    def add_function(self, func: Callable):
        """Add a function to the cluster."""
        self.functions.append(func)

    def add_local_code_path(self, path: str):
        """Add local code path to be bundled with cluster."""
        self.local_code_paths.append(path)

    def analyze_cluster_dependencies(
        self,
        analyzer: DependencyAnalyzer,
    ) -> dict[str, list[str]]:
        """Analyze dependencies for entire cluster."""
        all_requirements = set()
        all_local_imports = set()
        all_external_calls = set()

        for func in self.functions:
            deps = analyzer.analyze_function_dependencies(func)
            all_requirements.update(deps["pip_requirements"])
            all_local_imports.update(deps["local_imports"])
            all_external_calls.update(deps["external_calls"])

        return {
            "pip_requirements": list(all_requirements),
            "local_imports": list(all_local_imports),
            "external_calls": list(all_external_calls),
            "code_paths": self.local_code_paths,
        }


def validate_function_safety(
    func: Callable,
    analyzer: DependencyAnalyzer,
) -> dict[str, Union[list[str], dict[str, list[str]]]]:
    """
    Validate that a function is safe to bundle without external dependencies.

    Args:
        func: Function to validate
        analyzer: DependencyAnalyzer instance

    Returns:
        Dictionary with validation results and warnings
    """
    deps = analyzer.analyze_function_dependencies(func)

    warnings = []
    errors = []

    # Check for external calls that might not be bundled
    if deps["external_calls"]:
        warnings.append(f"External function calls detected: {deps['external_calls']}")
        warnings.append("These functions may not be available when transform is loaded")

    # Check for local imports that need to be bundled
    if deps["local_imports"]:
        warnings.append(f"Local imports detected: {deps['local_imports']}")
        warnings.append("These modules should be included in code_paths")

    # Check for complex third-party dependencies
    complex_deps = ["tensorflow", "torch", "sklearn", "xgboost", "lightgbm"]
    complex_found = [
        dep
        for dep in deps["pip_requirements"]
        if any(complex_dep in dep for complex_dep in complex_deps)
    ]

    if complex_found:
        warnings.append(f"Complex dependencies detected: {complex_found}")
        warnings.append("These may cause dependency conflicts with other transforms")

    return {"warnings": warnings, "errors": errors, "dependencies": deps}


def create_minimal_requirements(
    functions: Union[Callable, list[Callable]],
    extra_requirements: Optional[list[str]] = None,
    code_paths: Optional[list[str]] = None,
) -> dict[str, list[str]]:
    """
    Create minimal requirements specification for function(s).

    Args:
        functions: Single function or list of functions
        extra_requirements: Additional pip requirements to include
        code_paths: Local code paths to bundle

    Returns:
        Dictionary with requirements and code paths
    """
    if not isinstance(functions, list):
        functions = [functions]

    analyzer = DependencyAnalyzer()

    # Analyze all functions
    all_requirements = set()
    all_warnings = []

    for func in functions:
        validation = validate_function_safety(func, analyzer)
        deps = validation["dependencies"]
        if isinstance(deps, dict):
            all_requirements.update(deps["pip_requirements"])
        if isinstance(validation["warnings"], list):
            all_warnings.extend(validation["warnings"])

    # Add extra requirements
    if extra_requirements:
        all_requirements.update(extra_requirements)

    # Always include core PySpark requirements
    core_requirements = ["pyspark<4.0", "mlflow>=2.22.0"]
    all_requirements.update(core_requirements)

    result = {
        "pip_requirements": list(all_requirements),
        "code_paths": code_paths or [],
        "warnings": all_warnings,
    }

    return result
