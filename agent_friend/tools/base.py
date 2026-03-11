"""base.py — BaseTool abstract class for agent-friend tools."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List


class BaseTool(ABC):
    """Abstract base class for all agent-friend tools.

    Subclasses must implement:
      - name: str property
      - description: str property
      - definitions: returns list of tool definitions for the LLM
      - execute(tool_name, arguments): runs the tool and returns a string result
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier for this tool (e.g. "memory", "code")."""

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description of what this tool does."""

    @abstractmethod
    def definitions(self) -> List[Dict[str, Any]]:
        """Return list of tool definitions in Anthropic tool-use format.

        Each definition is a dict with:
          name: str
          description: str
          input_schema: dict (JSON Schema)
        """

    @abstractmethod
    def execute(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Execute a tool call and return the result as a string.

        Parameters
        ----------
        tool_name:  The specific function name called (from definitions).
        arguments:  Dict of argument name -> value.

        Returns
        -------
        String result to return to the LLM.
        """
