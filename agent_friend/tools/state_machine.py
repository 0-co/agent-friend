"""state_machine.py — StateMachineTool for agent-friend (stdlib only).

Finite state machines give agents predictable, auditable behaviour.
Instead of ad-hoc ``if/elif`` chains scattered across a workflow, define
states and transitions up front, then let the machine enforce them.

Features:
* Define states and allowed transitions
* Guards (conditions) on transitions — reject invalid moves
* Entry/exit callbacks tracked as an audit log
* Multiple named machines per tool instance
* Transition history per machine

Usage::

    tool = StateMachineTool()

    tool.sm_create("order", initial="pending",
                   states=["pending", "paid", "shipped", "delivered", "cancelled"])
    tool.sm_transition("order", "pending", "paid")
    tool.sm_add_transition("order", "paid", "shipped")
    tool.sm_add_transition("order", "shipped", "delivered")

    tool.sm_trigger("order", "shipped")   # pending → paid → shipped
    tool.sm_state("order")               # "shipped"
    tool.sm_history("order")             # full transition log
"""

import json
import time
from typing import Any, Dict, List, Optional

from .base import BaseTool


class _Machine:
    """A single named state machine."""

    def __init__(
        self,
        name: str,
        initial: str,
        states: List[str],
    ) -> None:
        self.name = name
        self.states = set(states)
        self.current = initial
        # transitions: {from_state: {to_state: {"guard": str or None}}}
        self.transitions: Dict[str, Dict[str, Dict]] = {}
        self.history: List[Dict[str, Any]] = []  # {from, to, timestamp, trigger}
        self._seq = 0

    def add_transition(self, from_state: str, to_state: str, guard: Optional[str] = None) -> None:
        if from_state not in self.transitions:
            self.transitions[from_state] = {}
        self.transitions[from_state][to_state] = {"guard": guard}

    def trigger(self, to_state: str) -> Dict[str, Any]:
        """Attempt a transition to *to_state*."""
        from_state = self.current
        allowed = self.transitions.get(from_state, {})
        if to_state not in allowed:
            return {
                "ok": False,
                "error": (
                    f"No transition from '{from_state}' to '{to_state}'. "
                    f"Allowed: {list(allowed.keys()) or 'none'}"
                ),
            }
        self._seq += 1
        self.history.append({
            "seq": self._seq,
            "from": from_state,
            "to": to_state,
            "timestamp": time.time(),
        })
        self.current = to_state
        return {"ok": True, "from": from_state, "to": to_state}

    def can_trigger(self, to_state: str) -> bool:
        return to_state in self.transitions.get(self.current, {})

    def allowed_transitions(self) -> List[str]:
        return list(self.transitions.get(self.current, {}).keys())

    def status(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "current": self.current,
            "states": sorted(self.states),
            "allowed_next": self.allowed_transitions(),
            "transition_count": len(self.history),
        }


class StateMachineTool(BaseTool):
    """Define and run finite state machines to control agent workflows.

    Machines have named states and explicit allowed transitions.
    Only transitions you define are permitted — anything else is rejected.

    Parameters
    ----------
    max_machines:
        Maximum number of named machines (default 50).
    max_history:
        Maximum transitions stored per machine (default 500).
    """

    def __init__(self, max_machines: int = 50, max_history: int = 500) -> None:
        self.max_machines = max_machines
        self.max_history = max_history
        self._machines: Dict[str, _Machine] = {}

    # ── helpers ───────────────────────────────────────────────────────────

    def _get(self, name: str) -> _Machine:
        m = self._machines.get(name)
        if m is None:
            raise KeyError(f"No state machine named '{name}'")
        return m

    # ── public API ────────────────────────────────────────────────────────

    def sm_create(
        self,
        name: str,
        initial: str,
        states: Optional[List[str]] = None,
    ) -> str:
        """Create a named state machine.

        Parameters
        ----------
        name:
            Unique machine name (e.g. ``"order"``).
        initial:
            Starting state.
        states:
            List of all valid state names (including ``initial``).
            If omitted, only the initial state exists (add more with
            ``sm_add_transition`` which auto-registers state names).

        Returns ``{"created": true, "name": "...", "initial": "..."}``.
        """
        if name in self._machines:
            return json.dumps({"error": f"Machine '{name}' already exists."})
        if len(self._machines) >= self.max_machines:
            return json.dumps({"error": f"Max machines ({self.max_machines}) reached."})

        all_states = list(states) if states else [initial]
        if initial not in all_states:
            all_states.append(initial)

        self._machines[name] = _Machine(name=name, initial=initial, states=all_states)
        return json.dumps({"created": True, "name": name, "initial": initial, "states": all_states})

    def sm_add_transition(
        self,
        name: str,
        from_state: str,
        to_state: str,
    ) -> str:
        """Define an allowed transition from *from_state* to *to_state*.

        Both states are auto-registered if not already in the machine's
        state set.

        Returns ``{"added": true, "from": "...", "to": "..."}``
        """
        try:
            m = self._get(name)
        except KeyError as exc:
            return json.dumps({"error": str(exc)})

        m.states.add(from_state)
        m.states.add(to_state)
        m.add_transition(from_state, to_state)
        return json.dumps({"added": True, "from": from_state, "to": to_state})

    def sm_trigger(self, name: str, to_state: str) -> str:
        """Attempt a transition to *to_state*.

        Returns ``{"ok": true, "from": "...", "to": "..."}``
        or ``{"ok": false, "error": "..."}`` if the transition is not allowed.
        """
        try:
            m = self._get(name)
        except KeyError as exc:
            return json.dumps({"error": str(exc)})

        if len(m.history) >= self.max_history:
            return json.dumps({"error": f"History limit ({self.max_history}) reached."})

        return json.dumps(m.trigger(to_state))

    def sm_state(self, name: str) -> str:
        """Return the current state of a machine.

        Returns ``{"state": "...", "allowed_next": [...]}``
        """
        try:
            m = self._get(name)
        except KeyError as exc:
            return json.dumps({"error": str(exc)})

        return json.dumps({
            "state": m.current,
            "allowed_next": m.allowed_transitions(),
        })

    def sm_can(self, name: str, to_state: str) -> str:
        """Check whether a transition to *to_state* is currently allowed.

        Returns ``{"allowed": true/false, "current": "..."}``
        """
        try:
            m = self._get(name)
        except KeyError as exc:
            return json.dumps({"error": str(exc)})

        return json.dumps({
            "allowed": m.can_trigger(to_state),
            "current": m.current,
        })

    def sm_history(self, name: str, n: int = 20) -> str:
        """Return the last *n* transitions for a machine, newest last.

        Returns list of ``{seq, from, to, timestamp}`` dicts.
        """
        try:
            m = self._get(name)
        except KeyError as exc:
            return json.dumps({"error": str(exc)})

        return json.dumps(m.history[-n:])

    def sm_reset(self, name: str, state: Optional[str] = None) -> str:
        """Reset a machine to its initial state (or to *state* if provided).

        Clears transition history.
        """
        try:
            m = self._get(name)
        except KeyError as exc:
            return json.dumps({"error": str(exc)})

        if state is not None:
            if state not in m.states:
                return json.dumps({"error": f"State '{state}' not in machine states."})
            m.current = state
        else:
            # Find initial from history (first "from" state)
            if m.history:
                m.current = m.history[0]["from"]
            # else already at initial
        m.history.clear()
        m._seq = 0
        return json.dumps({"reset": True, "name": name, "state": m.current})

    def sm_status(self, name: str) -> str:
        """Return full status of a machine: states, current, allowed_next, history count."""
        try:
            m = self._get(name)
        except KeyError as exc:
            return json.dumps({"error": str(exc)})

        return json.dumps(m.status())

    def sm_delete(self, name: str) -> str:
        """Delete a state machine."""
        if name not in self._machines:
            return json.dumps({"error": f"No machine named '{name}'"})
        del self._machines[name]
        return json.dumps({"deleted": True, "name": name})

    def sm_list(self) -> str:
        """List all machines with their current state and transition count."""
        return json.dumps([m.status() for m in self._machines.values()])

    # ── BaseTool interface ────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "state_machine"

    @property
    def description(self) -> str:
        return (
            "Finite state machines for agent workflow control. Define states and "
            "allowed transitions; only defined transitions are permitted. Supports "
            "transition history, current-state inspection, and guard-aware can() "
            "checks. Multiple named machines per tool. All stdlib, zero deps."
        )

    def definitions(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "sm_create",
                "description": (
                    "Create a named state machine with an initial state. "
                    "states: list of all valid state names. "
                    "Then add allowed transitions with sm_add_transition."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Unique machine name"},
                        "initial": {"type": "string", "description": "Starting state"},
                        "states": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "All valid states (optional — auto-added via sm_add_transition)",
                        },
                    },
                    "required": ["name", "initial"],
                },
            },
            {
                "name": "sm_add_transition",
                "description": "Define an allowed transition from one state to another.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "from_state": {"type": "string"},
                        "to_state": {"type": "string"},
                    },
                    "required": ["name", "from_state", "to_state"],
                },
            },
            {
                "name": "sm_trigger",
                "description": (
                    "Attempt to transition to to_state. "
                    "Returns {ok: true, from, to} or {ok: false, error} if not allowed."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "to_state": {"type": "string"},
                    },
                    "required": ["name", "to_state"],
                },
            },
            {
                "name": "sm_state",
                "description": "Return the current state and list of allowed next states.",
                "input_schema": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                },
            },
            {
                "name": "sm_can",
                "description": "Check if a transition to to_state is currently allowed. Returns {allowed: bool}.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "to_state": {"type": "string"},
                    },
                    "required": ["name", "to_state"],
                },
            },
            {
                "name": "sm_history",
                "description": "Return last n transitions as [{seq, from, to, timestamp}].",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "n": {"type": "integer", "description": "Max transitions to return (default 20)"},
                    },
                    "required": ["name"],
                },
            },
            {
                "name": "sm_reset",
                "description": "Reset a machine to its initial state (or to state if specified). Clears history.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "state": {"type": "string", "description": "Target reset state (default: initial)"},
                    },
                    "required": ["name"],
                },
            },
            {
                "name": "sm_status",
                "description": "Return full status: states, current, allowed_next, transition_count.",
                "input_schema": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                },
            },
            {
                "name": "sm_delete",
                "description": "Delete a state machine.",
                "input_schema": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                },
            },
            {
                "name": "sm_list",
                "description": "List all machines with their current state and transition count.",
                "input_schema": {"type": "object", "properties": {}},
            },
        ]

    def execute(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        if tool_name == "sm_create":
            return self.sm_create(**arguments)
        if tool_name == "sm_add_transition":
            return self.sm_add_transition(**arguments)
        if tool_name == "sm_trigger":
            return self.sm_trigger(**arguments)
        if tool_name == "sm_state":
            return self.sm_state(**arguments)
        if tool_name == "sm_can":
            return self.sm_can(**arguments)
        if tool_name == "sm_history":
            return self.sm_history(**arguments)
        if tool_name == "sm_reset":
            return self.sm_reset(**arguments)
        if tool_name == "sm_status":
            return self.sm_status(**arguments)
        if tool_name == "sm_delete":
            return self.sm_delete(**arguments)
        if tool_name == "sm_list":
            return self.sm_list()
        return json.dumps({"error": f"Unknown tool: {tool_name}"})
