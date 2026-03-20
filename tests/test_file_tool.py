"""Tests for FileTool."""

import os
import pathlib
import tempfile
import pytest

from agent_friend.tools.file import FileTool


@pytest.fixture
def tmp_tool(tmp_path):
    """FileTool rooted at a temp directory."""
    return FileTool(base_dir=str(tmp_path)), tmp_path


class TestFileRead:
    def test_read_existing_file(self, tmp_tool):
        tool, base = tmp_tool
        (base / "hello.txt").write_text("hello world")
        result = tool.execute("file_read", {"path": "hello.txt"})
        assert "hello world" in result

    def test_read_missing_file(self, tmp_tool):
        tool, _ = tmp_tool
        result = tool.execute("file_read", {"path": "nope.txt"})
        assert "not found" in result.lower()

    def test_read_directory_rejected(self, tmp_tool):
        tool, base = tmp_tool
        (base / "subdir").mkdir()
        result = tool.execute("file_read", {"path": "subdir"})
        assert "directory" in result.lower()

    def test_read_truncates_large_files(self, tmp_path):
        tool = FileTool(base_dir=str(tmp_path), max_read_bytes=10)
        (tmp_path / "big.txt").write_text("A" * 100)
        result = tool.execute("file_read", {"path": "big.txt"})
        assert "truncated" in result

    def test_read_outside_base_rejected(self, tmp_tool):
        tool, _ = tmp_tool
        result = tool.execute("file_read", {"path": "../etc/passwd"})
        assert "denied" in result.lower() or "outside" in result.lower()

    def test_read_utf8_content(self, tmp_tool):
        tool, base = tmp_tool
        (base / "unicode.txt").write_text("héllo wörld 🤖")
        result = tool.execute("file_read", {"path": "unicode.txt"})
        assert "héllo" in result


class TestFileWrite:
    def test_write_creates_file(self, tmp_tool):
        tool, base = tmp_tool
        result = tool.execute("file_write", {"path": "out.txt", "content": "test content"})
        assert "written" in result.lower()
        assert (base / "out.txt").read_text() == "test content"

    def test_write_overwrites_existing(self, tmp_tool):
        tool, base = tmp_tool
        (base / "f.txt").write_text("old")
        tool.execute("file_write", {"path": "f.txt", "content": "new"})
        assert (base / "f.txt").read_text() == "new"

    def test_write_creates_parent_dirs(self, tmp_tool):
        tool, base = tmp_tool
        result = tool.execute("file_write", {"path": "a/b/c.txt", "content": "deep"})
        assert "written" in result.lower()
        assert (base / "a" / "b" / "c.txt").read_text() == "deep"

    def test_write_outside_base_rejected(self, tmp_tool):
        tool, _ = tmp_tool
        result = tool.execute("file_write", {"path": "../evil.txt", "content": "x"})
        assert "denied" in result.lower() or "outside" in result.lower()

    def test_write_reports_byte_count(self, tmp_tool):
        tool, _ = tmp_tool
        result = tool.execute("file_write", {"path": "f.txt", "content": "12345"})
        assert "5" in result


class TestFileAppend:
    def test_append_to_existing(self, tmp_tool):
        tool, base = tmp_tool
        (base / "log.txt").write_text("line1\n")
        tool.execute("file_append", {"path": "log.txt", "content": "line2\n"})
        assert (base / "log.txt").read_text() == "line1\nline2\n"

    def test_append_creates_file(self, tmp_tool):
        tool, base = tmp_tool
        tool.execute("file_append", {"path": "new.txt", "content": "first"})
        assert (base / "new.txt").read_text() == "first"

    def test_append_outside_base_rejected(self, tmp_tool):
        tool, _ = tmp_tool
        result = tool.execute("file_append", {"path": "../x.txt", "content": "x"})
        assert "denied" in result.lower() or "outside" in result.lower()


class TestFileList:
    def test_list_directory(self, tmp_tool):
        tool, base = tmp_tool
        (base / "a.txt").write_text("a")
        (base / "b.txt").write_text("b")
        (base / "sub").mkdir()
        result = tool.execute("file_list", {"path": "."})
        assert "a.txt" in result
        assert "b.txt" in result
        assert "sub" in result

    def test_list_with_pattern(self, tmp_tool):
        tool, base = tmp_tool
        (base / "a.py").write_text("")
        (base / "b.txt").write_text("")
        result = tool.execute("file_list", {"path": ".", "pattern": "*.py"})
        assert "a.py" in result
        assert "b.txt" not in result

    def test_list_missing_path(self, tmp_tool):
        tool, _ = tmp_tool
        result = tool.execute("file_list", {"path": "missing_dir"})
        assert "not found" in result.lower()

    def test_list_file_rejected(self, tmp_tool):
        tool, base = tmp_tool
        (base / "f.txt").write_text("")
        result = tool.execute("file_list", {"path": "f.txt"})
        assert "file" in result.lower() or "directory" in result.lower()

    def test_list_empty_directory(self, tmp_tool):
        tool, base = tmp_tool
        (base / "empty").mkdir()
        result = tool.execute("file_list", {"path": "empty"})
        assert "empty" in result.lower()

    def test_list_outside_base_rejected(self, tmp_tool):
        tool, _ = tmp_tool
        result = tool.execute("file_list", {"path": "../.."})
        assert "denied" in result.lower() or "outside" in result.lower()


class TestFileToolIntegration:
    def test_unknown_tool_name(self, tmp_tool):
        tool, _ = tmp_tool
        result = tool.execute("file_unknown", {})
        assert "unknown" in result.lower()

    def test_tool_name_is_file(self, tmp_tool):
        tool, _ = tmp_tool
        assert tool.name == "file"

    def test_definitions_has_four_operations(self, tmp_tool):
        tool, _ = tmp_tool
        defs = tool.definitions()
        names = {d["name"] for d in defs}
        assert names == {"file_read", "file_write", "file_append", "file_list"}

    def test_friend_accepts_file_tool_name(self):
        """Friend can instantiate 'file' tool by string name."""
        from agent_friend import Friend
        friend = Friend(tools=["file"], seed="You are a helpful assistant.")
        assert any(t.name == "file" for t in friend._tools)

    def test_default_base_dir_is_cwd(self):
        tool = FileTool()
        assert tool.base_dir == pathlib.Path(os.getcwd()).resolve()
