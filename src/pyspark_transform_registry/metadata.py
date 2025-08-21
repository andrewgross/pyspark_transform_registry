def _resolve_fully_qualified_name(obj):
    """Resolve the fully qualified name of an object."""
    if obj is None:
        return None
    module = obj.__module__
    qualname = getattr(obj, "__qualname__", obj.__name__)
    return f"{module}.{qualname}"
