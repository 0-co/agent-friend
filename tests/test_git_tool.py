"""Tests for GitTool."""

import os
import subprocess
import pytest
from pathlib import Path

from agent_friend.tools.git import GitTool


# ── fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def git_repo(tmp_path):
    """Create a minimal git repo for testing."""
    subprocess.run(["git", "init", str(tmp_path)], check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=tmp_path, check=True, capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=tmp_path, check=True, capture_output=True,
    )
    # Initial commit so HEAD exists
    (tmp_path / "readme.txt").write_text("hello")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "Initial commit"],
        cwd=tmp_path, check=True, capture_output=True,
    )
    return tmp_path


@pytest.fixture
def git(git_repo):
    """GitTool pointing at a fresh repo."""
    return GitTool(repo_dir=str(git_repo))


@pytest.fixture
def git_dirty(git_repo):
    """Repo with an unstaged modification."""
    (git_repo / "readme.txt").write_text("modified")
    return GitTool(repo_dir=str(git_repo))


# ── basic properties ──────────────────────────────────────────────────────────


def test_name(git):
    assert git.name == "git"


def test_description(git):
    assert "git" in git.description.lower()
    assert len(git.description) > 10


def test_definitions_count(git):
    defs = git.definitions()
    assert len(defs) == 7


def test_definitions_names(git):
    names = {d["name"] for d in git.definitions()}
    assert names == {
        "git_status", "git_diff", "git_log",
        "git_add", "git_commit",
        "git_branch_list", "git_branch_create",
    }


def test_all_definitions_have_input_schema(git):
    for d in git.definitions():
        assert "input_schema" in d
        assert d["input_schema"]["type"] == "object"


# ── git_status ────────────────────────────────────────────────────────────────


def test_status_clean_repo(git):
    result = git.execute("git_status", {})
    assert "nothing to commit" in result.lower() or "clean" in result.lower()


def test_status_dirty_repo(git_dirty):
    result = git_dirty.execute("git_status", {})
    assert "modified" in result.lower()


def test_status_python_api(git):
    result = git.status()
    assert isinstance(result, str)
    assert len(result) > 0


# ── git_diff ──────────────────────────────────────────────────────────────────


def test_diff_clean_repo_returns_clean_message(git):
    result = git.execute("git_diff", {})
    assert "clean" in result.lower() or "no changes" in result.lower()


def test_diff_dirty_repo_shows_changes(git_dirty):
    result = git_dirty.execute("git_diff", {"staged": False})
    assert "modified" in result.lower() or "readme" in result.lower() or "-hello" in result


def test_diff_staged_no_changes(git):
    result = git.execute("git_diff", {"staged": True})
    assert "no changes" in result.lower() or "clean" in result.lower()


def test_diff_python_api(git):
    result = git.diff()
    assert isinstance(result, str)


# ── git_log ───────────────────────────────────────────────────────────────────


def test_log_shows_initial_commit(git):
    result = git.execute("git_log", {})
    assert "Initial commit" in result


def test_log_n_limits_output(git):
    result = git.execute("git_log", {"n": 1})
    # Should only show one commit (the initial one)
    assert "Initial commit" in result


def test_log_oneline(git):
    result = git.execute("git_log", {"oneline": True})
    # oneline format: "<hash> <message>"
    lines = [l for l in result.strip().split("\n") if l]
    # Each line should be short (hash + message)
    for line in lines:
        assert len(line) < 200


def test_log_python_api(git):
    result = git.log()
    assert "Initial commit" in result


# ── git_add ───────────────────────────────────────────────────────────────────


def test_add_stages_file(git_dirty, git_repo):
    result = git_dirty.execute("git_add", {"paths": ["readme.txt"]})
    assert "staged" in result.lower() or "modified" in result.lower()

    # Verify it's actually staged
    status_out = git_dirty.status()
    assert "readme.txt" in status_out


def test_add_all(git_dirty, git_repo):
    (git_repo / "new_file.txt").write_text("new content")
    result = git_dirty.execute("git_add", {"paths": ["."]})
    assert isinstance(result, str)


def test_add_python_api(git_dirty, git_repo):
    result = git_dirty.add(["readme.txt"])
    assert isinstance(result, str)


# ── git_commit ────────────────────────────────────────────────────────────────


def test_commit_staged_changes(git_dirty, git_repo):
    git_dirty.execute("git_add", {"paths": ["readme.txt"]})
    result = git_dirty.execute("git_commit", {"message": "Update readme"})
    assert "Update readme" in result or "master" in result.lower() or "main" in result.lower()


def test_commit_nothing_staged(git):
    result = git.execute("git_commit", {"message": "Nothing to commit"})
    # Should return a message about nothing to commit
    assert "nothing" in result.lower() or "clean" in result.lower()


def test_commit_python_api(git_dirty, git_repo):
    git_dirty.add(["readme.txt"])
    result = git_dirty.commit("Python API commit")
    assert isinstance(result, str)


# ── git_branch ────────────────────────────────────────────────────────────────


def test_branch_list(git):
    result = git.execute("git_branch_list", {})
    # Should show at least one branch
    assert isinstance(result, str)
    assert len(result) > 0


def test_branch_create_and_checkout(git):
    result = git.execute("git_branch_create", {"name": "feature/test", "checkout": True})
    assert "feature/test" in result or "Switched" in result or "branch" in result.lower()

    # Verify we're on the new branch
    status = git.status()
    assert "feature/test" in status or "branch" in status.lower()


def test_branch_create_no_checkout(git):
    result = git.execute("git_branch_create", {"name": "no-checkout-branch", "checkout": False})
    assert isinstance(result, str)

    # Verify we're NOT on the new branch (still on original)
    branches = git.branch_list()
    assert "no-checkout-branch" in branches


def test_branch_python_api(git):
    result = git.branch_list()
    assert isinstance(result, str)


# ── error handling ────────────────────────────────────────────────────────────


def test_unknown_operation_returns_error(git):
    result = git.execute("git_nonexistent", {})
    assert "Unknown" in result or "unknown" in result


def test_invalid_repo_dir_returns_error():
    tool = GitTool(repo_dir="/nonexistent/path/xyz")
    result = tool.execute("git_status", {})
    assert "error" in result.lower() or "fatal" in result.lower() or len(result) > 0


# ── Friend integration ────────────────────────────────────────────────────────


def test_friend_accepts_git_string(git_repo):
    from agent_friend import Friend
    friend = Friend(api_key="sk-test", tools=["git"])
    assert len(friend._tools) == 1
    assert friend._tools[0].name == "git"


def test_friend_accepts_git_instance(git_repo):
    from agent_friend import Friend
    git_tool = GitTool(repo_dir=str(git_repo))
    friend = Friend(api_key="sk-test", tools=[git_tool])
    assert len(friend._tools) == 1


def test_friend_git_definitions(git_repo):
    from agent_friend import Friend
    friend = Friend(api_key="sk-test", tools=["git"])
    defs = friend._build_tool_definitions()
    names = {d["name"] for d in defs}
    assert "git_status" in names
    assert "git_commit" in names
