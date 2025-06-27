import inspect
import pydoc
import typing
from typing import Callable


def validate_transform_input(func: Callable, input_obj) -> bool:
    """
    Validates that the first argument's type of a transform function matches the input object's type.
    """
    sig = inspect.signature(func)
    params = list(sig.parameters.values())
    if not params:
        return True  # no input to validate

    first_param = params[0].name
    hints = typing.get_type_hints(func)
    expected_type = hints.get(first_param)
    if expected_type is None:
        return True

    resolved = pydoc.locate(f"{expected_type.__module__}.{expected_type.__qualname__}")
    return isinstance(input_obj, resolved) if resolved and inspect.isclass(resolved) else False