"""function_tool.py — @tool decorator and FunctionTool for custom agent functions."""

import inspect
import types
import typing
from typing import Any, Callable, Dict, List, Optional, get_type_hints

from .base import BaseTool


# ---------------------------------------------------------------------------
# JSON Schema helpers
# ---------------------------------------------------------------------------

_PY_TYPE_TO_JSON: Dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
    bytes: "string",
}


def _is_optional(py_type: Any) -> bool:
    """Return True if py_type is Optional[X] (i.e. Union[X, None])."""
    origin = getattr(py_type, "__origin__", None)
    # typing.Optional[X] = Union[X, None]
    if origin is typing.Union:
        return type(None) in py_type.__args__
    # Python 3.10+ `X | None` syntax
    if hasattr(types, "UnionType") and isinstance(py_type, types.UnionType):
        return type(None) in py_type.__args__
    return False


def _unwrap_optional(py_type: Any) -> Any:
    """Extract the non-None type from Optional[X]."""
    args = [a for a in py_type.__args__ if a is not type(None)]
    return args[0] if args else str


def _python_type_to_json_schema(py_type: Any) -> Dict[str, Any]:
    """Convert a Python type annotation to a JSON Schema property dict."""
    if _is_optional(py_type):
        py_type = _unwrap_optional(py_type)
    json_type = _PY_TYPE_TO_JSON.get(py_type, "string")
    return {"type": json_type}


def _build_input_schema(fn: Callable) -> Dict[str, Any]:
    """Build a JSON Schema object for a function's parameters.

    Uses type hints for property types and default values / Optional to
    determine which parameters are required.
    """
    try:
        hints = get_type_hints(fn)
    except Exception:
        hints = {}

    sig = inspect.signature(fn)
    properties: Dict[str, Any] = {}
    required: List[str] = []

    for param_name, param in sig.parameters.items():
        if param_name == "self":
            continue
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue

        py_type = hints.get(param_name, str)
        is_opt = _is_optional(py_type)

        properties[param_name] = _python_type_to_json_schema(py_type)

        # Required when no default value and not Optional
        if param.default is inspect.Parameter.empty and not is_opt:
            required.append(param_name)

    schema: Dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema


# ---------------------------------------------------------------------------
# FunctionTool
# ---------------------------------------------------------------------------


class FunctionTool(BaseTool):
    """A BaseTool that wraps a plain Python function.

    Created automatically by the :func:`tool` decorator, but can also be
    instantiated directly:

        def my_fn(city: str) -> str:
            return f"Sunny in {city}"

        t = FunctionTool(my_fn, name="weather", description="Get weather")
        friend = Friend(tools=[t])
    """

    def __init__(
        self,
        fn: Callable,
        tool_name: str,
        tool_description: str,
        input_schema: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._fn = fn
        self._tool_name = tool_name
        self._tool_description = tool_description
        self._input_schema = input_schema or _build_input_schema(fn)

    @property
    def name(self) -> str:
        return self._tool_name

    @property
    def description(self) -> str:
        return self._tool_description

    def definitions(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": self._tool_name,
                "description": self._tool_description,
                "input_schema": self._input_schema,
            }
        ]

    def execute(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        result = self._fn(**arguments)
        if isinstance(result, str):
            return result
        return str(result)


# ---------------------------------------------------------------------------
# @tool decorator
# ---------------------------------------------------------------------------


def tool(
    fn: Optional[Callable] = None,
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
) -> Any:
    """Register a Python function as an agent tool.

    The decorated function remains fully callable normally — only a
    ``_agent_tool`` attribute is added so that :class:`Friend` can detect it.

    Usage::

        from agent_friend import Friend, tool

        @tool
        def get_weather(city: str) -> str:
            \"\"\"Get current weather for a city.\"\"\"
            return f"Sunny in {city}"

        @tool(name="add", description="Add two numbers")
        def add_numbers(a: int, b: int) -> int:
            return a + b

        friend = Friend(tools=["search", get_weather, add_numbers])

        # Functions are still callable normally
        print(get_weather("London"))   # "Sunny in London"
        print(add_numbers(2, 3))       # 5

    Parameters
    ----------
    fn:
        The function to wrap (when used as ``@tool`` without parentheses).
    name:
        Override the tool name. Defaults to the function's ``__name__``.
    description:
        Override the tool description. Defaults to the function's docstring.
    """

    def decorator(f: Callable) -> Callable:
        tool_name = name or f.__name__
        tool_desc = description or (f.__doc__ or "").strip() or f.__name__
        f._agent_tool = FunctionTool(f, tool_name, tool_desc)
        return f

    if fn is not None:
        # Used as @tool without parentheses
        return decorator(fn)
    return decorator
