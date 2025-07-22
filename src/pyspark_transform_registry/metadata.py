import inspect
import textwrap
import typing
from collections.abc import Callable


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
    return_type: str | None,
):
    """
    Creates a wrapped version of the function's source code with parameter and return type metadata
    and docstring embedded as a header comment.

    Note: This function is now primarily used for backwards compatibility.
    MLflow model signatures provide better metadata handling for the model registry.
    """
    # Dedent the source to remove any indentation from nested function definitions
    dedented_source = textwrap.dedent(source)

    # Add necessary imports for PySpark functions
    imports = """# Required imports for PySpark transforms
from pyspark.sql import DataFrame
from pyspark.sql.functions import *

"""

    # Build parameters section
    params_section = ""
    for p in param_info:
        annotation = f" ({p['annotation']})" if p["annotation"] else ""
        default = f", default={p['default']}" if p["default"] is not None else ""
        params_section += f"        #   - {p['name']}{annotation}{default}\n"

    # Build docstring section
    doc_section = ""
    if doc:
        for line in doc.split("\n"):
            doc_section += f"        # {line}\n"

    # Build complete header with natural indentation
    header = f"""        # Auto-logged transform function: {name}
        #
        # Args:
{params_section}        #
        # Returns: {return_type or "unspecified"}
        #
{doc_section}        #

        """

    # Apply dedent to remove common leading whitespace
    header = textwrap.dedent(header)
    return f"{imports}{header}{dedented_source}"


def extract_metadata_from_signature(signature) -> dict:
    """
    Extract metadata from an MLflow model signature.

    Args:
        signature: MLflow ModelSignature object

    Returns:
        Dictionary containing extracted metadata
    """
    metadata = {}

    if signature and signature.inputs:
        metadata["input_schema"] = [
            {
                "name": col.name,
                "type": str(col.type),
                "optional": getattr(col, "optional", False),
            }
            for col in signature.inputs.inputs
        ]

    if signature and signature.outputs:
        metadata["output_schema"] = [
            {
                "name": col.name,
                "type": str(col.type),
                "optional": getattr(col, "optional", False),
            }
            for col in signature.outputs.inputs
        ]

    return metadata
