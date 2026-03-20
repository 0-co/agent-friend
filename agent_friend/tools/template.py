"""template.py — TemplateTool for agent-friend (stdlib only).

Agents that hardcode prompts are brittle. TemplateTool gives your agent
reusable, parameterized templates — with variable substitution, validation,
and file-based template management.

Uses Python's stdlib string.Template (${var} syntax) for safe substitution.
No extra dependencies.

Usage::

    tool = TemplateTool()
    tool.template_render("Hello, ${name}!", {"name": "agent"})
    # "Hello, agent!"

    tool.template_save("greeting", "Hello, ${name}! You have ${count} messages.")
    tool.template_render_named("greeting", {"name": "Alice", "count": 3})
    # "Hello, Alice! You have 3 messages."

    tool.template_variables("Hello, ${name}! Today is ${date}.")
    # ["name", "date"]
"""

import json
import re
import string
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import BaseTool


class TemplateTool(BaseTool):
    """Parameterized string templates for agent prompts and content.

    Supports ${variable} substitution (Python string.Template syntax).
    Templates can be stored in-memory by name and reused across calls.
    All operations are stdlib-only — zero dependencies.
    """

    def __init__(self, template_dir: Optional[str] = None) -> None:
        # In-memory named templates
        self._templates: Dict[str, str] = {}
        # Optional disk persistence directory
        self._template_dir: Optional[Path] = Path(template_dir) if template_dir else None

    @property
    def name(self) -> str:
        return "template"

    @property
    def description(self) -> str:
        return (
            "Render parameterized string templates using ${variable} syntax. "
            "Save named templates for reuse, validate variables before rendering, "
            "and render from strings or named templates."
        )

    def definitions(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "template_render",
                "description": (
                    "Render a template string with the given variables. "
                    "Uses ${variable} syntax (Python string.Template). "
                    "Returns the rendered string or an error if variables are missing."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "template": {"type": "string", "description": "Template string with ${variable} placeholders."},
                        "variables": {
                            "type": "object",
                            "description": "Key-value pairs for substitution (e.g. {name: 'Alice', count: 5}).",
                        },
                        "strict": {
                            "type": "boolean",
                            "description": "If true (default), error on missing variables. If false, leave them as-is.",
                        },
                    },
                    "required": ["template", "variables"],
                },
            },
            {
                "name": "template_save",
                "description": (
                    "Save a named template for later reuse. "
                    "Templates are stored in-memory for the session. "
                    "Returns the template name and variable list."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Name for the template (e.g. 'search_prompt')."},
                        "template": {"type": "string", "description": "Template string with ${variable} placeholders."},
                    },
                    "required": ["name", "template"],
                },
            },
            {
                "name": "template_render_named",
                "description": (
                    "Render a previously saved named template with the given variables. "
                    "Returns the rendered string or an error if the template doesn't exist."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Template name (saved with template_save)."},
                        "variables": {
                            "type": "object",
                            "description": "Key-value pairs for substitution.",
                        },
                        "strict": {
                            "type": "boolean",
                            "description": "If true (default), error on missing variables.",
                        },
                    },
                    "required": ["name", "variables"],
                },
            },
            {
                "name": "template_variables",
                "description": (
                    "Extract the list of variable names from a template string. "
                    "Returns the sorted list of ${variable} names found."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "template": {"type": "string", "description": "Template string to inspect."},
                    },
                    "required": ["template"],
                },
            },
            {
                "name": "template_validate",
                "description": (
                    "Check that a set of variables satisfies a template's requirements. "
                    "Returns {valid: bool, missing: [...], extra: [...]}."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "template": {"type": "string", "description": "Template string to validate against."},
                        "variables": {
                            "type": "object",
                            "description": "Variables to check.",
                        },
                    },
                    "required": ["template", "variables"],
                },
            },
            {
                "name": "template_list",
                "description": (
                    "List all saved named templates. "
                    "Returns [{name, variables: [...]}, ...] sorted by name."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {},
                },
            },
            {
                "name": "template_get",
                "description": (
                    "Get a saved named template's source string and variable list. "
                    "Returns {name, template, variables} or an error if not found."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Template name."},
                    },
                    "required": ["name"],
                },
            },
            {
                "name": "template_delete",
                "description": (
                    "Delete a saved named template. "
                    "Returns {deleted: name} or an error if not found."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Template name to delete."},
                    },
                    "required": ["name"],
                },
            },
        ]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_variables(template_str: str) -> List[str]:
        """Extract ${variable} names from a template string (sorted, deduplicated)."""
        # Match ${identifier} and $identifier patterns
        pattern = re.compile(r'\$\{([_a-zA-Z][_a-zA-Z0-9]*)\}|\$([_a-zA-Z][_a-zA-Z0-9]*)')
        found = set()
        for m in pattern.finditer(template_str):
            name = m.group(1) or m.group(2)
            found.add(name)
        return sorted(found)

    @staticmethod
    def _render(template_str: str, variables: Dict[str, Any], strict: bool = True) -> str:
        """Render a template. Raises KeyError on missing vars if strict=True."""
        # Convert all variable values to strings
        str_vars = {k: str(v) for k, v in variables.items()}
        tmpl = string.Template(template_str)
        if strict:
            return tmpl.substitute(str_vars)
        else:
            return tmpl.safe_substitute(str_vars)

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def template_render(
        self,
        template: str,
        variables: Dict[str, Any],
        strict: bool = True,
    ) -> Dict[str, Any]:
        try:
            result = self._render(template, variables, strict)
            return {"rendered": result, "length": len(result)}
        except (KeyError, ValueError) as exc:
            return {"error": f"Template rendering failed: {exc}"}

    def template_save(self, name: str, template: str) -> Dict[str, Any]:
        if not name.strip():
            return {"error": "Template name cannot be empty"}
        self._templates[name] = template
        variables = self._extract_variables(template)
        return {"name": name, "saved": True, "variables": variables}

    def template_render_named(
        self,
        name: str,
        variables: Dict[str, Any],
        strict: bool = True,
    ) -> Dict[str, Any]:
        if name not in self._templates:
            return {"error": f"Template '{name}' not found. Use template_save to create it."}
        return self.template_render(self._templates[name], variables, strict)

    def template_variables(self, template: str) -> Dict[str, Any]:
        variables = self._extract_variables(template)
        return {"variables": variables, "count": len(variables)}

    def template_validate(
        self,
        template: str,
        variables: Dict[str, Any],
    ) -> Dict[str, Any]:
        required = set(self._extract_variables(template))
        provided = set(variables.keys())
        missing = sorted(required - provided)
        extra = sorted(provided - required)
        return {
            "valid": len(missing) == 0,
            "missing": missing,
            "extra": extra,
            "required": sorted(required),
            "provided": sorted(provided),
        }

    def template_list(self) -> List[Dict[str, Any]]:
        return sorted(
            [
                {"name": n, "variables": self._extract_variables(t)}
                for n, t in self._templates.items()
            ],
            key=lambda x: x["name"],
        )

    def template_get(self, name: str) -> Dict[str, Any]:
        if name not in self._templates:
            return {"error": f"Template '{name}' not found"}
        tmpl = self._templates[name]
        return {
            "name": name,
            "template": tmpl,
            "variables": self._extract_variables(tmpl),
        }

    def template_delete(self, name: str) -> Dict[str, Any]:
        if name not in self._templates:
            return {"error": f"Template '{name}' not found"}
        del self._templates[name]
        return {"deleted": name}

    # ------------------------------------------------------------------
    # execute
    # ------------------------------------------------------------------

    def execute(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        try:
            if tool_name == "template_render":
                return json.dumps(
                    self.template_render(
                        arguments["template"],
                        arguments.get("variables", {}),
                        bool(arguments.get("strict", True)),
                    )
                )

            elif tool_name == "template_save":
                return json.dumps(
                    self.template_save(arguments["name"], arguments["template"])
                )

            elif tool_name == "template_render_named":
                return json.dumps(
                    self.template_render_named(
                        arguments["name"],
                        arguments.get("variables", {}),
                        bool(arguments.get("strict", True)),
                    )
                )

            elif tool_name == "template_variables":
                return json.dumps(self.template_variables(arguments["template"]))

            elif tool_name == "template_validate":
                return json.dumps(
                    self.template_validate(
                        arguments["template"],
                        arguments.get("variables", {}),
                    )
                )

            elif tool_name == "template_list":
                return json.dumps(self.template_list())

            elif tool_name == "template_get":
                return json.dumps(self.template_get(arguments["name"]))

            elif tool_name == "template_delete":
                return json.dumps(self.template_delete(arguments["name"]))

            else:
                return json.dumps({"error": f"Unknown tool: {tool_name}"})

        except (KeyError, ValueError, TypeError) as exc:
            return json.dumps({"error": str(exc)})
