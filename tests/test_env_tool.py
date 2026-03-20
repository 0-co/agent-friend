"""Tests for EnvTool."""

import json
import os
import pytest

from agent_friend.tools.env import EnvTool


# ── fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def env():
    """Return an unrestricted EnvTool."""
    return EnvTool()


@pytest.fixture
def clean_env(monkeypatch):
    """Remove test variables before each test."""
    for k in ["TEST_FOO", "TEST_BAR", "TEST_BAZ"]:
        monkeypatch.delenv(k, raising=False)
    yield


# ── basic properties ──────────────────────────────────────────────────────────


def test_name(env):
    assert env.name == "env"


def test_description(env):
    assert "environment" in env.description.lower()


def test_definitions_returns_five_tools(env):
    defs = env.definitions()
    assert len(defs) == 5
    names = {d["name"] for d in defs}
    assert names == {"env_get", "env_set", "env_list", "env_check", "env_load"}


# ── env_get ───────────────────────────────────────────────────────────────────


def test_env_get_existing(env, monkeypatch):
    monkeypatch.setenv("TEST_FOO", "hello")
    assert env.env_get("TEST_FOO") == "hello"


def test_env_get_missing_returns_null(env, clean_env):
    assert env.env_get("TEST_FOO") == "null"


def test_env_get_missing_with_default(env, clean_env):
    assert env.env_get("TEST_FOO", "fallback") == "fallback"


def test_env_get_sensitive_hidden(env):
    """Variables matching SECRET/TOKEN/KEY pattern are hidden."""
    result = env.env_get("SOME_API_KEY")
    assert result == "[hidden — use env_check to verify it is set]"


def test_env_get_sensitive_hidden_token(env):
    result = env.env_get("GITHUB_TOKEN")
    assert "[hidden" in result


def test_env_get_allow_sensitive(monkeypatch):
    """allow_sensitive=True bypasses the block."""
    monkeypatch.setenv("MY_API_KEY", "abc123")
    tool = EnvTool(allow_sensitive=True)
    assert tool.env_get("MY_API_KEY") == "abc123"


# ── env_set ───────────────────────────────────────────────────────────────────


def test_env_set_roundtrip(env, monkeypatch):
    monkeypatch.delenv("TEST_BAR", raising=False)
    result = env.env_set("TEST_BAR", "world")
    assert "TEST_BAR" in result
    assert os.environ.get("TEST_BAR") == "world"


def test_env_set_returns_confirmation(env, monkeypatch):
    monkeypatch.delenv("TEST_BAZ", raising=False)
    out = env.env_set("TEST_BAZ", "42")
    assert "TEST_BAZ" in out
    assert "42" in out


# ── env_list ──────────────────────────────────────────────────────────────────


def test_env_list_returns_json(env):
    data = json.loads(env.env_list())
    assert isinstance(data, dict)


def test_env_list_prefix_filter(env, monkeypatch):
    monkeypatch.setenv("TEST_FOO", "1")
    monkeypatch.setenv("TEST_BAR", "2")
    data = json.loads(env.env_list(prefix="TEST_"))
    assert "TEST_FOO" in data
    assert "TEST_BAR" in data


def test_env_list_prefix_excludes_others(env, monkeypatch):
    monkeypatch.setenv("MY_VAR", "yes")
    monkeypatch.setenv("TEST_FOO", "x")
    data = json.loads(env.env_list(prefix="TEST_"))
    assert "MY_VAR" not in data
    assert "TEST_FOO" in data


def test_env_list_excludes_sensitive_by_default(env, monkeypatch):
    monkeypatch.setenv("MY_API_KEY", "secret123")
    data = json.loads(env.env_list())
    assert "MY_API_KEY" not in data


def test_env_list_safe_prefixes(monkeypatch):
    monkeypatch.setenv("HOME", "/home/user")
    monkeypatch.setenv("USER", "alice")
    monkeypatch.setenv("PATH", "/usr/bin")
    tool = EnvTool(safe_prefixes=["HO"])
    data = json.loads(tool.env_list())
    assert "HOME" in data
    assert "USER" not in data


# ── env_check ─────────────────────────────────────────────────────────────────


def test_env_check_all_present(env, monkeypatch):
    monkeypatch.setenv("TEST_FOO", "a")
    monkeypatch.setenv("TEST_BAR", "b")
    result = json.loads(env.env_check(["TEST_FOO", "TEST_BAR"]))
    assert result["ok"] is True
    assert "TEST_FOO" in result["present"]
    assert "TEST_BAR" in result["present"]
    assert result["missing"] == []


def test_env_check_some_missing(env, monkeypatch):
    monkeypatch.setenv("TEST_FOO", "a")
    monkeypatch.delenv("TEST_BAR", raising=False)
    result = json.loads(env.env_check(["TEST_FOO", "TEST_BAR"]))
    assert result["ok"] is False
    assert "TEST_FOO" in result["present"]
    assert "TEST_BAR" in result["missing"]


def test_env_check_all_missing(env, clean_env):
    result = json.loads(env.env_check(["TEST_FOO", "TEST_BAR", "TEST_BAZ"]))
    assert result["ok"] is False
    assert result["present"] == []
    assert set(result["missing"]) == {"TEST_FOO", "TEST_BAR", "TEST_BAZ"}


def test_env_check_empty_list(env):
    result = json.loads(env.env_check([]))
    assert result["ok"] is True
    assert result["present"] == []
    assert result["missing"] == []


def test_env_check_works_for_sensitive_vars(env, monkeypatch):
    """env_check can verify sensitive vars are set without revealing them."""
    monkeypatch.setenv("MY_SECRET_KEY", "s3cr3t")
    result = json.loads(env.env_check(["MY_SECRET_KEY"]))
    assert result["ok"] is True
    assert "MY_SECRET_KEY" in result["present"]


# ── env_load ──────────────────────────────────────────────────────────────────


def test_env_load_basic(env, tmp_path, monkeypatch):
    dotenv = tmp_path / ".env"
    dotenv.write_text("TEST_FOO=bar\nTEST_BAR=baz\n")
    monkeypatch.delenv("TEST_FOO", raising=False)
    monkeypatch.delenv("TEST_BAR", raising=False)
    result = json.loads(env.env_load(str(dotenv)))
    assert "TEST_FOO" in result["loaded"]
    assert "TEST_BAR" in result["loaded"]
    assert result["errors"] == []
    assert os.environ.get("TEST_FOO") == "bar"
    assert os.environ.get("TEST_BAR") == "baz"


def test_env_load_quoted_values(env, tmp_path, monkeypatch):
    dotenv = tmp_path / ".env"
    dotenv.write_text('TEST_FOO="hello world"\nTEST_BAR=\'single\'\n')
    monkeypatch.delenv("TEST_FOO", raising=False)
    monkeypatch.delenv("TEST_BAR", raising=False)
    env.env_load(str(dotenv))
    assert os.environ.get("TEST_FOO") == "hello world"
    assert os.environ.get("TEST_BAR") == "single"


def test_env_load_skips_comments(env, tmp_path, monkeypatch):
    dotenv = tmp_path / ".env"
    dotenv.write_text("# this is a comment\nTEST_FOO=yes\n")
    monkeypatch.delenv("TEST_FOO", raising=False)
    result = json.loads(env.env_load(str(dotenv)))
    assert "TEST_FOO" in result["loaded"]
    assert result["errors"] == []


def test_env_load_does_not_overwrite(env, tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_FOO", "original")
    dotenv = tmp_path / ".env"
    dotenv.write_text("TEST_FOO=new_value\n")
    result = json.loads(env.env_load(str(dotenv)))
    assert "TEST_FOO" in result["skipped"]
    assert os.environ.get("TEST_FOO") == "original"


def test_env_load_file_not_found(env):
    result = json.loads(env.env_load("/nonexistent/.env"))
    assert "error" in result


def test_env_load_empty_file(env, tmp_path, monkeypatch):
    dotenv = tmp_path / ".env"
    dotenv.write_text("")
    result = json.loads(env.env_load(str(dotenv)))
    assert result["loaded"] == []
    assert result["skipped"] == []
    assert result["errors"] == []


def test_env_load_skips_empty_lines(env, tmp_path, monkeypatch):
    dotenv = tmp_path / ".env"
    dotenv.write_text("\n\nTEST_FOO=yes\n\n")
    monkeypatch.delenv("TEST_FOO", raising=False)
    result = json.loads(env.env_load(str(dotenv)))
    assert "TEST_FOO" in result["loaded"]


def test_env_load_invalid_line_reported(env, tmp_path):
    dotenv = tmp_path / ".env"
    dotenv.write_text("THIS_HAS_NO_EQUALS\n")
    result = json.loads(env.env_load(str(dotenv)))
    assert len(result["errors"]) == 1


# ── execute dispatch ──────────────────────────────────────────────────────────


def test_execute_env_get(env, monkeypatch):
    monkeypatch.setenv("TEST_FOO", "xyz")
    assert env.execute("env_get", {"key": "TEST_FOO"}) == "xyz"


def test_execute_env_set(env, monkeypatch):
    monkeypatch.delenv("TEST_SET", raising=False)
    env.execute("env_set", {"key": "TEST_SET", "value": "done"})
    assert os.environ.get("TEST_SET") == "done"


def test_execute_env_list(env):
    result = json.loads(env.execute("env_list", {}))
    assert isinstance(result, dict)


def test_execute_env_list_with_prefix(env, monkeypatch):
    monkeypatch.setenv("TEST_FOO", "1")
    result = json.loads(env.execute("env_list", {"prefix": "TEST_"}))
    assert "TEST_FOO" in result


def test_execute_env_check(env, monkeypatch):
    monkeypatch.setenv("TEST_FOO", "1")
    result = json.loads(env.execute("env_check", {"keys": ["TEST_FOO"]}))
    assert result["ok"] is True


def test_execute_env_load(env, tmp_path, monkeypatch):
    dotenv = tmp_path / ".env"
    dotenv.write_text("TEST_FOO=loaded\n")
    monkeypatch.delenv("TEST_FOO", raising=False)
    env.execute("env_load", {"path": str(dotenv)})
    assert os.environ.get("TEST_FOO") == "loaded"


def test_execute_unknown(env):
    result = env.execute("env_unknown", {})
    assert "Unknown" in result
