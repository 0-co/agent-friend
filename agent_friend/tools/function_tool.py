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


def _parse_docstring_params(fn: Callable) -> Dict[str, str]:
    """Extract parameter descriptions from Google-style docstrings.

    Parses the ``Args:`` section of a docstring and returns a mapping of
    parameter name to description text.

    Example::

        def my_func(city: str, unit: str = "celsius") -> dict:
            \"\"\"Get weather for a city.

            Args:
                city: The city name
                unit: Temperature unit (celsius or fahrenheit)
            \"\"\"

        _parse_docstring_params(my_func)
        # => {"city": "The city name", "unit": "Temperature unit (celsius or fahrenheit)"}
    """
    doc = inspect.getdoc(fn)
    if not doc:
        return {}

    params: Dict[str, str] = {}
    lines = doc.splitlines()
    in_args = False
    current_param: Optional[str] = None
    current_desc: List[str] = []

    for line in lines:
        stripped = line.strip()
        # Detect start of Args section
        if stripped in ("Args:", "Arguments:", "Parameters:", "Params:"):
            in_args = True
            continue
        # Detect end of Args section (another section header or blank after content)
        if in_args and stripped and stripped.endswith(":") and ":" not in stripped[:-1]:
            # This is a new section header like "Returns:" or "Raises:"
            # Save current param if any
            if current_param is not None:
                params[current_param] = " ".join(current_desc).strip()
            in_args = False
            continue
        if not in_args:
            continue
        # Inside Args section
        if not stripped:
            # Blank line might end the section or just be spacing
            continue
        # Check if this is a new parameter line (param_name: description)
        if ":" in stripped:
            # Could be "param_name: desc" or "param_name (type): desc"
            colon_idx = stripped.index(":")
            candidate = stripped[:colon_idx].strip()
            # Remove optional type annotation like "param_name (str)"
            paren_idx = candidate.find("(")
            if paren_idx > 0:
                candidate = candidate[:paren_idx].strip()
            # Validate it looks like a parameter name (identifier, no spaces)
            if candidate.isidentifier():
                # Save previous param
                if current_param is not None:
                    params[current_param] = " ".join(current_desc).strip()
                current_param = candidate
                current_desc = [stripped[colon_idx + 1:].strip()]
                continue
        # Continuation line for current parameter
        if current_param is not None:
            current_desc.append(stripped)

    # Save the last parameter
    if current_param is not None:
        params[current_param] = " ".join(current_desc).strip()

    return params


def _build_input_schema(fn: Callable) -> Dict[str, Any]:
    """Build a JSON Schema object for a function's parameters.

    Uses type hints for property types and default values / Optional to
    determine which parameters are required.  Docstring parameter
    descriptions (Google-style ``Args:`` section) are included when present.
    """
    try:
        hints = get_type_hints(fn)
    except Exception:
        hints = {}

    sig = inspect.signature(fn)
    properties: Dict[str, Any] = {}
    required: List[str] = []
    doc_params = _parse_docstring_params(fn)

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

        prop = _python_type_to_json_schema(py_type)
        if param_name in doc_params:
            prop["description"] = doc_params[param_name]
        properties[param_name] = prop

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
        ft = FunctionTool(f, tool_name, tool_desc)
        f._agent_tool = ft
        # Proxy adapter methods for convenient access
        f.to_anthropic = ft.to_anthropic
        f.to_openai = ft.to_openai
        f.to_google = ft.to_google
        f.to_mcp = ft.to_mcp
        f.to_json_schema = ft.to_json_schema
        f.definitions = ft.definitions
        return f

    if fn is not None:
        # Used as @tool without parentheses
        return decorator(fn)
    return decorator
