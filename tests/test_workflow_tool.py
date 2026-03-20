"""Tests for WorkflowTool — lightweight workflow / pipeline runner."""

import json
import pytest
from agent_friend.tools.workflow_tool import WorkflowTool


@pytest.fixture
def tool():
    return WorkflowTool()


@pytest.fixture
def wf(tool):
    """Tool with a simple 'etl' workflow pre-defined."""
    tool.workflow_define("etl", steps=[
        {"name": "strip", "fn": "strip"},
        {"name": "upper", "fn": "upper"},
    ])
    return tool


# ── workflow_define ─────────────────────────────────────────────────────────

def test_define_basic(tool):
    r = json.loads(tool.workflow_define("pipe", steps=[{"name": "s1", "fn": "identity"}]))
    assert r["workflow"] == "pipe"
    assert r["step_count"] == 1


def test_define_multiple_steps(tool):
    steps = [{"fn": "strip"}, {"fn": "upper"}, {"fn": "lower"}]
    r = json.loads(tool.workflow_define("multi", steps=steps))
    assert r["step_count"] == 3


def test_define_empty_name_error(tool):
    r = json.loads(tool.workflow_define("", steps=[{"fn": "identity"}]))
    assert "error" in r


def test_define_empty_steps_error(tool):
    r = json.loads(tool.workflow_define("x", steps=[]))
    assert "error" in r


def test_define_non_list_steps_error(tool):
    r = json.loads(tool.workflow_define("x", steps="not a list"))
    assert "error" in r


def test_define_step_fn_not_string_error(tool):
    r = json.loads(tool.workflow_define("x", steps=[{"fn": 123}]))
    assert "error" in r


def test_define_too_many_retries_error(tool):
    r = json.loads(tool.workflow_define("x", steps=[{"fn": "identity", "retries": 99}]))
    assert "error" in r


def test_define_invalid_on_error(tool):
    r = json.loads(tool.workflow_define("x", steps=[{"fn": "identity", "on_error": "explode"}]))
    assert "error" in r


def test_define_invalid_condition(tool):
    r = json.loads(tool.workflow_define("x", steps=[{"fn": "identity", "condition": "maybe"}]))
    assert "error" in r


def test_define_overwrite(tool):
    tool.workflow_define("w", steps=[{"fn": "upper"}])
    r = json.loads(tool.workflow_define("w", steps=[{"fn": "lower"}, {"fn": "strip"}]))
    assert r["step_count"] == 2


# ── workflow_run ────────────────────────────────────────────────────────────

def test_run_identity(tool):
    tool.workflow_define("id", steps=[{"fn": "identity"}])
    r = json.loads(tool.workflow_run("id", input=42))
    assert r["output"] == 42
    assert r["ok"] is True


def test_run_upper(tool):
    tool.workflow_define("up", steps=[{"fn": "upper"}])
    r = json.loads(tool.workflow_run("up", input="hello"))
    assert r["output"] == "HELLO"


def test_run_chained(wf):
    r = json.loads(wf.workflow_run("etl", input="  hello world  "))
    assert r["output"] == "HELLO WORLD"


def test_run_step_results(wf):
    r = json.loads(wf.workflow_run("etl", input="  hi  "))
    assert len(r["steps"]) == 2
    assert r["steps"][0]["name"] == "strip"
    assert r["steps"][1]["name"] == "upper"
    assert r["steps"][0]["status"] == "ok"


def test_run_not_found_error(tool):
    r = json.loads(tool.workflow_run("nope", input="x"))
    assert "error" in r


def test_run_unknown_fn_fails(tool):
    tool.workflow_define("bad", steps=[{"fn": "does_not_exist"}])
    r = json.loads(tool.workflow_run("bad", input="x"))
    assert r["ok"] is False
    assert r["steps"][0]["status"] == "failed"


def test_run_on_error_skip(tool):
    tool.workflow_define("skip_wf", steps=[
        {"fn": "does_not_exist", "on_error": "skip"},
        {"fn": "upper"},
    ])
    r = json.loads(tool.workflow_run("skip_wf", input="hello"))
    assert r["ok"] is True
    assert r["output"] == "HELLO"


def test_run_on_error_default(tool):
    tool.workflow_define("def_wf", steps=[
        {"fn": "does_not_exist", "on_error": "default", "default": "FALLBACK"},
        {"fn": "lower"},
    ])
    r = json.loads(tool.workflow_run("def_wf", input="anything"))
    assert r["output"] == "fallback"


def test_run_elapsed_ms_present(wf):
    r = json.loads(wf.workflow_run("etl", input="x"))
    assert "elapsed_ms" in r
    assert isinstance(r["elapsed_ms"], float)


def test_run_increments_run_count(tool):
    tool.workflow_define("rc", steps=[{"fn": "identity"}])
    tool.workflow_run("rc", input=1)
    tool.workflow_run("rc", input=2)
    wf_info = json.loads(tool.workflow_get("rc"))
    assert wf_info["run_count"] == 2


def test_run_with_context(tool):
    tool.workflow_define("ctx_wf", steps=[{"fn": "identity"}])
    r = json.loads(tool.workflow_run("ctx_wf", input="val", context={"key": "v"}))
    assert r["ok"] is True


def test_run_none_input(tool):
    tool.workflow_define("none_wf", steps=[{"fn": "identity"}])
    r = json.loads(tool.workflow_run("none_wf", input=None))
    assert r["output"] is None


# ── builtin functions ───────────────────────────────────────────────────────

def test_builtin_upper(tool):
    tool.workflow_define("t", steps=[{"fn": "upper"}])
    assert json.loads(tool.workflow_run("t", input="abc"))["output"] == "ABC"


def test_builtin_lower(tool):
    tool.workflow_define("t", steps=[{"fn": "lower"}])
    assert json.loads(tool.workflow_run("t", input="ABC"))["output"] == "abc"


def test_builtin_strip(tool):
    tool.workflow_define("t", steps=[{"fn": "strip"}])
    assert json.loads(tool.workflow_run("t", input="  hi  "))["output"] == "hi"


def test_builtin_to_int(tool):
    tool.workflow_define("t", steps=[{"fn": "to_int"}])
    assert json.loads(tool.workflow_run("t", input="42"))["output"] == 42


def test_builtin_to_float(tool):
    tool.workflow_define("t", steps=[{"fn": "to_float"}])
    assert json.loads(tool.workflow_run("t", input="3.14"))["output"] == pytest.approx(3.14)


def test_builtin_to_str(tool):
    tool.workflow_define("t", steps=[{"fn": "to_str"}])
    assert json.loads(tool.workflow_run("t", input=99))["output"] == "99"


def test_builtin_to_list_string(tool):
    tool.workflow_define("t", steps=[{"fn": "to_list"}])
    r = json.loads(tool.workflow_run("t", input="abc"))
    assert r["output"] == ["a", "b", "c"]


def test_builtin_reverse_string(tool):
    tool.workflow_define("t", steps=[{"fn": "reverse"}])
    assert json.loads(tool.workflow_run("t", input="hello"))["output"] == "olleh"


def test_builtin_reverse_list(tool):
    tool.workflow_define("t", steps=[{"fn": "reverse"}])
    assert json.loads(tool.workflow_run("t", input=[1, 2, 3]))["output"] == [3, 2, 1]


def test_builtin_length(tool):
    tool.workflow_define("t", steps=[{"fn": "length"}])
    assert json.loads(tool.workflow_run("t", input=[1, 2, 3]))["output"] == 3


def test_builtin_sum_list(tool):
    tool.workflow_define("t", steps=[{"fn": "sum_list"}])
    assert json.loads(tool.workflow_run("t", input=[1, 2, 3]))["output"] == 6


def test_builtin_sort(tool):
    tool.workflow_define("t", steps=[{"fn": "sort"}])
    assert json.loads(tool.workflow_run("t", input=[3, 1, 2]))["output"] == [1, 2, 3]


def test_builtin_unique(tool):
    tool.workflow_define("t", steps=[{"fn": "unique"}])
    assert json.loads(tool.workflow_run("t", input=[1, 2, 1, 3, 2]))["output"] == [1, 2, 3]


def test_builtin_flatten(tool):
    tool.workflow_define("t", steps=[{"fn": "flatten"}])
    assert json.loads(tool.workflow_run("t", input=[[1, 2], [3], 4]))["output"] == [1, 2, 3, 4]


def test_builtin_noop(tool):
    tool.workflow_define("t", steps=[{"fn": "noop"}])
    assert json.loads(tool.workflow_run("t", input=99))["output"] == 99


# ── condition ───────────────────────────────────────────────────────────────

def test_condition_truthy_runs(tool):
    tool.workflow_define("cond", steps=[
        {"fn": "upper", "condition": "truthy"},
    ])
    r = json.loads(tool.workflow_run("cond", input="hello"))
    assert r["output"] == "HELLO"


def test_condition_truthy_skips_on_falsy_input(tool):
    tool.workflow_define("cond", steps=[
        {"fn": "upper", "condition": "truthy"},
    ])
    r = json.loads(tool.workflow_run("cond", input=""))
    assert r["steps"][0]["status"] == "skipped_condition"
    assert r["output"] == ""


def test_condition_falsy_runs_on_empty(tool):
    tool.workflow_define("cond", steps=[
        {"fn": "noop", "condition": "falsy"},
    ])
    r = json.loads(tool.workflow_run("cond", input=""))
    assert r["steps"][0]["status"] == "ok"


def test_condition_falsy_skips_on_truthy(tool):
    tool.workflow_define("cond", steps=[
        {"fn": "upper", "condition": "falsy"},
    ])
    r = json.loads(tool.workflow_run("cond", input="hello"))
    assert r["steps"][0]["status"] == "skipped_condition"


# ── retries ─────────────────────────────────────────────────────────────────

def test_retries_on_bad_fn_exhausted(tool):
    # With retries, still fails after exhausting
    tool.workflow_define("retry_wf", steps=[{"fn": "to_int", "retries": 2}])
    r = json.loads(tool.workflow_run("retry_wf", input="not_a_number"))
    assert r["ok"] is False


# ── step_define ─────────────────────────────────────────────────────────────

def test_step_define_basic(tool):
    r = json.loads(tool.step_define("double", "def step(value, ctx): return value * 2"))
    assert r["registered"] is True


def test_step_define_used_in_workflow(tool):
    tool.step_define("double", "def step(value, ctx): return value * 2")
    tool.workflow_define("dbl_wf", steps=[{"fn": "double"}])
    r = json.loads(tool.workflow_run("dbl_wf", input=5))
    assert r["output"] == 10


def test_step_define_context_access(tool):
    tool.step_define("add_n", "def step(value, ctx): return value + ctx.get('n', 0)")
    tool.workflow_define("add_wf", steps=[{"fn": "add_n"}])
    r = json.loads(tool.workflow_run("add_wf", input=10, context={"n": 5}))
    assert r["output"] == 15


def test_step_define_empty_name_error(tool):
    r = json.loads(tool.step_define("", "def step(v, c): return v"))
    assert "error" in r


def test_step_define_no_step_fn_error(tool):
    r = json.loads(tool.step_define("bad", "def other(v, c): return v"))
    assert "error" in r


def test_step_define_syntax_error(tool):
    r = json.loads(tool.step_define("broken", "def step(v c): return v"))
    assert "error" in r


def test_step_define_conflicts_builtin(tool):
    r = json.loads(tool.step_define("upper", "def step(v, c): return v"))
    assert "error" in r


# ── workflow_list ───────────────────────────────────────────────────────────

def test_list_empty(tool):
    r = json.loads(tool.workflow_list())
    assert r["count"] == 0
    assert r["workflows"] == []


def test_list_after_define(tool):
    tool.workflow_define("a", steps=[{"fn": "identity"}])
    tool.workflow_define("b", steps=[{"fn": "upper"}])
    r = json.loads(tool.workflow_list())
    assert r["count"] == 2
    names = {w["name"] for w in r["workflows"]}
    assert "a" in names and "b" in names


# ── workflow_get ────────────────────────────────────────────────────────────

def test_get_basic(tool):
    tool.workflow_define("mywf", steps=[{"name": "s", "fn": "upper"}], description="test")
    r = json.loads(tool.workflow_get("mywf"))
    assert r["name"] == "mywf"
    assert r["description"] == "test"
    assert len(r["steps"]) == 1
    assert r["steps"][0]["fn"] == "upper"


def test_get_not_found(tool):
    r = json.loads(tool.workflow_get("missing"))
    assert "error" in r


# ── workflow_delete ─────────────────────────────────────────────────────────

def test_delete_basic(tool):
    tool.workflow_define("del_wf", steps=[{"fn": "identity"}])
    r = json.loads(tool.workflow_delete("del_wf"))
    assert r["deleted"] is True
    assert json.loads(tool.workflow_list())["count"] == 0


def test_delete_not_found(tool):
    r = json.loads(tool.workflow_delete("ghost"))
    assert "error" in r


# ── workflow_status ─────────────────────────────────────────────────────────

def test_status_empty(tool):
    r = json.loads(tool.workflow_status())
    assert r["total_runs"] == 0
    assert r["ok_runs"] == 0


def test_status_after_runs(tool):
    tool.workflow_define("s", steps=[{"fn": "identity"}])
    tool.workflow_run("s", input=1)
    tool.workflow_run("s", input=2)
    r = json.loads(tool.workflow_status())
    assert r["total_runs"] == 2
    assert r["ok_runs"] == 2
    assert r["failed_runs"] == 0


def test_status_counts_failures(tool):
    tool.workflow_define("fail_wf", steps=[{"fn": "unknown_fn"}])
    tool.workflow_run("fail_wf", input="x")
    r = json.loads(tool.workflow_status())
    assert r["failed_runs"] == 1


# ── builtin_fns ─────────────────────────────────────────────────────────────

def test_builtin_fns_list(tool):
    r = json.loads(tool.builtin_fns())
    assert "functions" in r
    assert "upper" in r["functions"]
    assert "identity" in r["functions"]
    assert r["count"] >= 10


# ── execute dispatch ─────────────────────────────────────────────────────────

def test_execute_define(tool):
    r = json.loads(tool.execute("workflow_define", {"name": "ex", "steps": [{"fn": "upper"}]}))
    assert "workflow" in r


def test_execute_run(tool):
    tool.workflow_define("ex", steps=[{"fn": "upper"}])
    r = json.loads(tool.execute("workflow_run", {"name": "ex", "input": "hi"}))
    assert r["output"] == "HI"


def test_execute_list(tool):
    r = json.loads(tool.execute("workflow_list", {}))
    assert "workflows" in r


def test_execute_status(tool):
    r = json.loads(tool.execute("workflow_status", {}))
    assert "total_runs" in r


def test_execute_builtin_fns(tool):
    r = json.loads(tool.execute("builtin_fns", {}))
    assert "functions" in r


def test_execute_unknown(tool):
    r = json.loads(tool.execute("nope", {}))
    assert "error" in r


# ── tool metadata ────────────────────────────────────────────────────────────

def test_name(tool):
    assert tool.name == "workflow"


def test_description(tool):
    assert "workflow" in tool.description.lower()


def test_definitions_count(tool):
    assert len(tool.definitions()) == 8


def test_definitions_fields(tool):
    for d in tool.definitions():
        assert "name" in d
        assert "description" in d
        assert "input_schema" in d
