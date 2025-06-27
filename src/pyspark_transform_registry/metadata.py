import inspect
import textwrap
import typing
from typing import Callable, Optional


def _resolve_fully_qualified_name(obj):
    """Resolve the fully qualified name of an object."""
    if obj is None:
        return None
    module = obj.__module__
    qualname = getattr(obj, "__qualname__", obj.__name__)
    return f"{module}.{qualname}"


def _get_function_metadata(func: Callable):
    """
    Extracts parameter information, return type annotation, and docstring from a function.
    """
    sig = inspect.signature(func)
    hints = typing.get_type_hints(func)
    param_info = []
    for name, param in sig.parameters.items():
        annotation = hints.get(name)
        param_info.append(
            {
                "name": name,
                "annotation": _resolve_fully_qualified_name(annotation)
                if annotation
                else None,
                "default": param.default if param.default != inspect._empty else None,
            },
        )
    return_type = hints.get("return")
    return_annot = _resolve_fully_qualified_name(return_type) if return_type else None
    doc = inspect.getdoc(func) or ""
    return param_info, return_annot, doc


def _wrap_function_source(
    name: str,
    source: str,
    doc: str,
    param_info,
    return_type: Optional[str],
):
    """
    Creates a wrapped version of the function's source code with parameter and return type metadata
    and docstring embedded as a header comment.
    """
    # Dedent the source to remove any indentation from nested function definitions
    dedented_source = textwrap.dedent(source)
    
    # Add necessary imports for PySpark functions
    imports = """# Required imports for PySpark transforms
from pyspark.sql import DataFrame
from pyspark.sql.functions import *

"""
    
    header = f"""# Auto-logged transform function: {name}
#
# Args:"""
    for p in param_info:
        annotation = f" ({p['annotation']})" if p["annotation"] else ""
        default = f", default={p['default']}" if p["default"] is not None else ""
        header += f"\n#   - {p['name']}{annotation}{default}"
    header += f"\n#\n# Returns: {return_type or 'unspecified'}\n#\n"
    if doc:
        # Add docstring as comments, line by line
        for line in doc.split('\n'):
            header += f"# {line}\n"
    header += "#\n\n"
    return f"{imports}{header}{dedented_source}"