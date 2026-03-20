"""Tests for StateMachineTool."""

import json
import pytest

from agent_friend.tools.state_machine import StateMachineTool


@pytest.fixture
def tool():
    return StateMachineTool()


@pytest.fixture
def order_machine(tool):
    """A simple order FSM: pending → paid → shipped → delivered (+ cancel from any)."""
    tool.sm_create("order", initial="pending",
                   states=["pending", "paid", "shipped", "delivered", "cancelled"])
    tool.sm_add_transition("order", "pending", "paid")
    tool.sm_add_transition("order", "pending", "cancelled")
    tool.sm_add_transition("order", "paid", "shipped")
    tool.sm_add_transition("order", "paid", "cancelled")
    tool.sm_add_transition("order", "shipped", "delivered")
    return tool


# ── basic properties ───────────────────────────────────────────────────────────


def test_name(tool):
    assert tool.name == "state_machine"


def test_description(tool):
    desc = tool.description.lower()
    assert "state" in desc


def test_definitions_count(tool):
    assert len(tool.definitions()) == 10


def test_definition_names(tool):
    names = {d["name"] for d in tool.definitions()}
    assert names == {
        "sm_create", "sm_add_transition", "sm_trigger",
        "sm_state", "sm_can", "sm_history", "sm_reset",
        "sm_status", "sm_delete", "sm_list",
    }


# ── sm_create ─────────────────────────────────────────────────────────────────


def test_create_basic(tool):
    result = json.loads(tool.sm_create("m", initial="idle"))
    assert result["created"] is True
    assert result["initial"] == "idle"


def test_create_with_states(tool):
    result = json.loads(tool.sm_create("m2", initial="a", states=["a", "b", "c"]))
    assert set(result["states"]) >= {"a", "b", "c"}


def test_create_duplicate_fails(tool):
    tool.sm_create("m", initial="idle")
    result = json.loads(tool.sm_create("m", initial="idle"))
    assert "error" in result


def test_create_initial_auto_added_to_states(tool):
    result = json.loads(tool.sm_create("m3", initial="start", states=["other"]))
    assert "start" in result["states"]


def test_create_max_machines():
    t = StateMachineTool(max_machines=2)
    t.sm_create("a", initial="x")
    t.sm_create("b", initial="x")
    result = json.loads(t.sm_create("c", initial="x"))
    assert "error" in result


# ── sm_add_transition ─────────────────────────────────────────────────────────


def test_add_transition(tool):
    tool.sm_create("m", initial="a")
    result = json.loads(tool.sm_add_transition("m", "a", "b"))
    assert result["added"] is True


def test_add_transition_registers_states(tool):
    tool.sm_create("m", initial="x")
    tool.sm_add_transition("m", "x", "y")
    status = json.loads(tool.sm_status("m"))
    assert "y" in status["states"]


def test_add_transition_unknown_machine(tool):
    result = json.loads(tool.sm_add_transition("ghost", "a", "b"))
    assert "error" in result


# ── sm_trigger ────────────────────────────────────────────────────────────────


def test_trigger_allowed(order_machine):
    result = json.loads(order_machine.sm_trigger("order", "paid"))
    assert result["ok"] is True
    assert result["from"] == "pending"
    assert result["to"] == "paid"


def test_trigger_denied(order_machine):
    result = json.loads(order_machine.sm_trigger("order", "delivered"))
    assert result["ok"] is False
    assert "error" in result


def test_trigger_updates_current(order_machine):
    order_machine.sm_trigger("order", "paid")
    state = json.loads(order_machine.sm_state("order"))
    assert state["state"] == "paid"


def test_trigger_chain(order_machine):
    order_machine.sm_trigger("order", "paid")
    order_machine.sm_trigger("order", "shipped")
    result = json.loads(order_machine.sm_trigger("order", "delivered"))
    assert result["ok"] is True
    assert result["to"] == "delivered"


def test_trigger_unknown_machine(tool):
    result = json.loads(tool.sm_trigger("ghost", "x"))
    assert "error" in result


def test_trigger_max_history():
    t = StateMachineTool(max_history=2)
    t.sm_create("m", initial="a")
    t.sm_add_transition("m", "a", "b")
    t.sm_add_transition("m", "b", "a")
    t.sm_trigger("m", "b")
    t.sm_trigger("m", "a")
    result = json.loads(t.sm_trigger("m", "b"))
    assert "error" in result


# ── sm_state ──────────────────────────────────────────────────────────────────


def test_state_initial(tool):
    tool.sm_create("m", initial="idle")
    result = json.loads(tool.sm_state("m"))
    assert result["state"] == "idle"


def test_state_allowed_next(order_machine):
    state = json.loads(order_machine.sm_state("order"))
    assert set(state["allowed_next"]) == {"paid", "cancelled"}


def test_state_unknown_machine(tool):
    result = json.loads(tool.sm_state("ghost"))
    assert "error" in result


# ── sm_can ────────────────────────────────────────────────────────────────────


def test_can_allowed(order_machine):
    result = json.loads(order_machine.sm_can("order", "paid"))
    assert result["allowed"] is True


def test_can_not_allowed(order_machine):
    result = json.loads(order_machine.sm_can("order", "shipped"))
    assert result["allowed"] is False


def test_can_after_transition(order_machine):
    order_machine.sm_trigger("order", "paid")
    result = json.loads(order_machine.sm_can("order", "shipped"))
    assert result["allowed"] is True


def test_can_unknown_machine(tool):
    result = json.loads(tool.sm_can("ghost", "x"))
    assert "error" in result


# ── sm_history ────────────────────────────────────────────────────────────────


def test_history_empty(tool):
    tool.sm_create("m", initial="x")
    result = json.loads(tool.sm_history("m"))
    assert result == []


def test_history_records_transitions(order_machine):
    order_machine.sm_trigger("order", "paid")
    order_machine.sm_trigger("order", "shipped")
    history = json.loads(order_machine.sm_history("order"))
    assert len(history) == 2
    assert history[0]["from"] == "pending"
    assert history[0]["to"] == "paid"
    assert history[1]["from"] == "paid"
    assert history[1]["to"] == "shipped"


def test_history_n_limit(order_machine):
    order_machine.sm_trigger("order", "paid")
    order_machine.sm_trigger("order", "shipped")
    history = json.loads(order_machine.sm_history("order", n=1))
    assert len(history) == 1
    assert history[0]["to"] == "shipped"


def test_history_has_timestamp(order_machine):
    order_machine.sm_trigger("order", "paid")
    history = json.loads(order_machine.sm_history("order"))
    assert "timestamp" in history[0]
    assert history[0]["timestamp"] > 0


def test_history_seq_increments(order_machine):
    order_machine.sm_trigger("order", "paid")
    order_machine.sm_trigger("order", "shipped")
    history = json.loads(order_machine.sm_history("order"))
    assert history[0]["seq"] == 1
    assert history[1]["seq"] == 2


# ── sm_reset ──────────────────────────────────────────────────────────────────


def test_reset_returns_to_initial(order_machine):
    order_machine.sm_trigger("order", "paid")
    order_machine.sm_reset("order")
    state = json.loads(order_machine.sm_state("order"))
    assert state["state"] == "pending"


def test_reset_clears_history(order_machine):
    order_machine.sm_trigger("order", "paid")
    order_machine.sm_reset("order")
    history = json.loads(order_machine.sm_history("order"))
    assert history == []


def test_reset_to_specific_state(order_machine):
    order_machine.sm_reset("order", state="shipped")
    state = json.loads(order_machine.sm_state("order"))
    assert state["state"] == "shipped"


def test_reset_invalid_state(tool):
    tool.sm_create("m", initial="x")
    result = json.loads(tool.sm_reset("m", state="nonexistent"))
    assert "error" in result


# ── sm_status ─────────────────────────────────────────────────────────────────


def test_status(order_machine):
    status = json.loads(order_machine.sm_status("order"))
    assert status["current"] == "pending"
    assert "pending" in status["states"]
    assert status["transition_count"] == 0


def test_status_after_triggers(order_machine):
    order_machine.sm_trigger("order", "paid")
    status = json.loads(order_machine.sm_status("order"))
    assert status["current"] == "paid"
    assert status["transition_count"] == 1


# ── sm_delete ─────────────────────────────────────────────────────────────────


def test_delete(tool):
    tool.sm_create("m", initial="x")
    tool.sm_delete("m")
    result = json.loads(tool.sm_state("m"))
    assert "error" in result


def test_delete_unknown(tool):
    result = json.loads(tool.sm_delete("ghost"))
    assert "error" in result


# ── sm_list ────────────────────────────────────────────────────────────────────


def test_list_empty(tool):
    result = json.loads(tool.sm_list())
    assert result == []


def test_list_shows_all(tool):
    tool.sm_create("a", initial="x")
    tool.sm_create("b", initial="y")
    result = json.loads(tool.sm_list())
    names = {m["name"] for m in result}
    assert names == {"a", "b"}


# ── execute dispatch ───────────────────────────────────────────────────────────


def test_execute_create(tool):
    result = json.loads(tool.execute("sm_create", {"name": "m", "initial": "idle"}))
    assert result["created"] is True


def test_execute_add_transition(tool):
    tool.execute("sm_create", {"name": "m", "initial": "a"})
    result = json.loads(tool.execute("sm_add_transition", {"name": "m", "from_state": "a", "to_state": "b"}))
    assert result["added"] is True


def test_execute_trigger(tool):
    tool.execute("sm_create", {"name": "m", "initial": "a"})
    tool.execute("sm_add_transition", {"name": "m", "from_state": "a", "to_state": "b"})
    result = json.loads(tool.execute("sm_trigger", {"name": "m", "to_state": "b"}))
    assert result["ok"] is True


def test_execute_state(tool):
    tool.execute("sm_create", {"name": "m", "initial": "idle"})
    result = json.loads(tool.execute("sm_state", {"name": "m"}))
    assert result["state"] == "idle"


def test_execute_can(tool):
    tool.execute("sm_create", {"name": "m", "initial": "a"})
    tool.execute("sm_add_transition", {"name": "m", "from_state": "a", "to_state": "b"})
    result = json.loads(tool.execute("sm_can", {"name": "m", "to_state": "b"}))
    assert result["allowed"] is True


def test_execute_history(tool):
    tool.execute("sm_create", {"name": "m", "initial": "a"})
    result = json.loads(tool.execute("sm_history", {"name": "m"}))
    assert result == []


def test_execute_reset(tool):
    tool.execute("sm_create", {"name": "m", "initial": "x"})
    result = json.loads(tool.execute("sm_reset", {"name": "m"}))
    assert result["reset"] is True


def test_execute_status(tool):
    tool.execute("sm_create", {"name": "m", "initial": "x"})
    result = json.loads(tool.execute("sm_status", {"name": "m"}))
    assert "current" in result


def test_execute_delete(tool):
    tool.execute("sm_create", {"name": "m", "initial": "x"})
    result = json.loads(tool.execute("sm_delete", {"name": "m"}))
    assert result["deleted"] is True


def test_execute_list(tool):
    result = json.loads(tool.execute("sm_list", {}))
    assert isinstance(result, list)


def test_execute_unknown(tool):
    result = json.loads(tool.execute("no_such", {}))
    assert "error" in result
