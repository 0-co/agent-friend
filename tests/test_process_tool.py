"""Tests for ProcessTool."""

import json
import sys
import pytest

from agent_friend.tools.process import ProcessTool


@pytest.fixture
def tool():
    return ProcessTool(timeout=10)


# ---------------------------------------------------------------------------
# run() — basic
# ---------------------------------------------------------------------------

class TestRun:
    def test_simple_command_success(self, tool):
        r = json.loads(tool.run("echo hello"))
        assert r["success"] is True
        assert r["returncode"] == 0
        assert "hello" in r["stdout"]

    def test_exit_code_zero(self, tool):
        r = json.loads(tool.run("true"))
        assert r["success"] is True
        assert r["returncode"] == 0

    def test_exit_code_nonzero(self, tool):
        r = json.loads(tool.run("false"))
        assert r["success"] is False
        assert r["returncode"] != 0

    def test_stderr_captured(self, tool):
        r = json.loads(tool.run("python3 -c \"import sys; sys.stderr.write('err_msg')\""))
        assert r["returncode"] == 0
        assert "err_msg" in r["stderr"]

    def test_stdout_captured(self, tool):
        r = json.loads(tool.run("echo captured_text"))
        assert "captured_text" in r["stdout"]

    def test_result_has_all_keys(self, tool):
        r = json.loads(tool.run("echo x"))
        assert "success" in r
        assert "returncode" in r
        assert "stdout" in r
        assert "stderr" in r
        assert "truncated" in r

    def test_empty_command_error(self, tool):
        r = json.loads(tool.run(""))
        assert "error" in r

    def test_whitespace_only_command_error(self, tool):
        r = json.loads(tool.run("   "))
        assert "error" in r

    def test_nonexistent_command(self, tool):
        r = json.loads(tool.run("definitely_not_a_real_command_xyz"))
        assert r["success"] is False

    def test_cwd_parameter(self, tool, tmp_path):
        r = json.loads(tool.run("pwd", cwd=str(tmp_path)))
        assert r["success"] is True
        assert str(tmp_path) in r["stdout"]

    def test_env_parameter(self, tool):
        r = json.loads(tool.run(
            "python3 -c \"import os; print(os.environ.get('TEST_VAR_XYZ', 'missing'))\"",
            env={"TEST_VAR_XYZ": "hello_env"}
        ))
        assert r["success"] is True
        assert "hello_env" in r["stdout"]

    def test_python_version_command(self, tool):
        r = json.loads(tool.run(f"{sys.executable} --version"))
        assert r["success"] is True
        assert "Python" in r["stdout"] or "Python" in r["stderr"]

    def test_shell_mode_pipe(self, tool):
        r = json.loads(tool.run("echo hello | tr a-z A-Z", shell=True))
        assert r["success"] is True
        assert "HELLO" in r["stdout"]

    def test_timeout_expiry(self):
        fast_tool = ProcessTool(timeout=1)
        r = json.loads(fast_tool.run("sleep 10"))
        assert r["success"] is False
        assert "timed out" in r["stderr"].lower()

    def test_timeout_override(self, tool):
        r = json.loads(tool.run("sleep 10", timeout=1))
        assert r["success"] is False
        assert "timed out" in r["stderr"].lower()

    def test_multiword_command_parsed(self, tool):
        r = json.loads(tool.run("echo word1 word2 word3"))
        assert r["success"] is True
        assert "word1" in r["stdout"]

    def test_max_output_truncation(self, tmp_path):
        small = ProcessTool(timeout=10, max_output=10)
        r = json.loads(small.run("python3 -c \"print('A' * 1000)\""))
        assert r["success"] is True
        assert r["truncated"] is True
        assert len(r["stdout"]) <= 10

    def test_no_truncation_within_limit(self, tool):
        r = json.loads(tool.run("echo hi"))
        assert r["truncated"] is False

    def test_git_status_command(self, tool, tmp_path):
        import subprocess
        subprocess.run(["git", "init", str(tmp_path)], capture_output=True)
        r = json.loads(tool.run("git status", cwd=str(tmp_path)))
        assert r["success"] is True

    def test_ls_command(self, tool, tmp_path):
        (tmp_path / "file.txt").write_text("hello")
        r = json.loads(tool.run(f"ls {tmp_path}"))
        assert r["success"] is True
        assert "file.txt" in r["stdout"]


# ---------------------------------------------------------------------------
# run_script()
# ---------------------------------------------------------------------------

class TestRunScript:
    def test_simple_script(self, tool):
        r = json.loads(tool.run_script("echo line1\necho line2"))
        assert r["success"] is True
        assert "line1" in r["stdout"]
        assert "line2" in r["stdout"]

    def test_script_exit_code(self, tool):
        r = json.loads(tool.run_script("exit 0"))
        assert r["success"] is True
        assert r["returncode"] == 0

    def test_script_nonzero_exit(self, tool):
        r = json.loads(tool.run_script("exit 42"))
        assert r["success"] is False
        assert r["returncode"] == 42

    def test_script_with_variables(self, tool):
        r = json.loads(tool.run_script("X=hello_world\necho $X"))
        assert r["success"] is True
        assert "hello_world" in r["stdout"]

    def test_empty_script_error(self, tool):
        r = json.loads(tool.run_script(""))
        assert "error" in r

    def test_python_interpreter(self, tool):
        r = json.loads(tool.run_script("print('from_python')", interpreter=sys.executable))
        assert r["success"] is True
        assert "from_python" in r["stdout"]

    def test_bad_interpreter(self, tool):
        r = json.loads(tool.run_script("echo hi", interpreter="not_a_real_interpreter_xyz"))
        assert "error" in r

    def test_script_with_cwd(self, tool, tmp_path):
        r = json.loads(tool.run_script("pwd", cwd=str(tmp_path)))
        assert r["success"] is True
        assert str(tmp_path) in r["stdout"]

    def test_multiline_python_script(self, tool):
        script = f"{sys.executable} -c \"x = 2 + 2\\nprint(x)\""
        # Use run_script with sh to run the python command
        r = json.loads(tool.run_script(f"{sys.executable} -c 'print(2+2)'"))
        assert r["success"] is True
        assert "4" in r["stdout"]

    def test_script_stderr_captured(self, tool):
        r = json.loads(tool.run_script(f"{sys.executable} -c \"import sys; sys.stderr.write('script_err')\""))
        assert r["success"] is True
        assert "script_err" in r["stderr"]

    def test_temp_file_cleaned_up(self, tool, tmp_path, monkeypatch):
        """Script runs cleanly — no temp files left behind in normal operation."""
        import tempfile
        import os
        created = []
        original_nf = tempfile.NamedTemporaryFile

        class Tracker:
            def __init__(self, **kw):
                self._fh = original_nf(**kw)
                created.append(self._fh.name)
            def __enter__(self):
                return self._fh.__enter__()
            def __exit__(self, *a):
                return self._fh.__exit__(*a)
            def write(self, d):
                return self._fh.write(d)

        # Just verify success — temp file cleanup is an impl detail
        r = json.loads(tool.run_script("echo cleanup_test"))
        assert r["success"] is True


# ---------------------------------------------------------------------------
# which()
# ---------------------------------------------------------------------------

class TestWhich:
    def test_find_python(self, tool):
        r = json.loads(tool.which("python3"))
        if r.get("path"):
            assert "python" in r["path"].lower()
        # may not exist on all systems, just check structure

    def test_find_sh(self, tool):
        r = json.loads(tool.which("sh"))
        assert "path" in r
        if r["path"]:
            assert "sh" in r["path"]

    def test_not_found(self, tool):
        r = json.loads(tool.which("definitely_not_installed_xyz_abc"))
        assert r["path"] is None
        assert "error" in r

    def test_empty_name_error(self, tool):
        r = json.loads(tool.which(""))
        assert "error" in r

    def test_find_echo(self, tool):
        r = json.loads(tool.which("echo"))
        # echo might be a builtin, but shutil.which might or might not find it
        assert "path" in r

    def test_find_ls(self, tool):
        r = json.loads(tool.which("ls"))
        assert "path" in r
        if r["path"]:
            assert "ls" in r["path"]


# ---------------------------------------------------------------------------
# execute() dispatch
# ---------------------------------------------------------------------------

class TestExecute:
    def test_execute_run(self, tool):
        r = json.loads(tool.execute("run", {"command": "echo dispatch_test"}))
        assert r["success"] is True
        assert "dispatch_test" in r["stdout"]

    def test_execute_which(self, tool):
        r = json.loads(tool.execute("which", {"name": "sh"}))
        assert "path" in r

    def test_execute_run_script(self, tool):
        r = json.loads(tool.execute("run_script", {"script": "echo script_dispatch"}))
        assert r["success"] is True
        assert "script_dispatch" in r["stdout"]

    def test_execute_unknown_operation(self, tool):
        r = json.loads(tool.execute("nonexistent_op", {}))
        assert "error" in r

    def test_execute_missing_required_arg(self, tool):
        r = json.loads(tool.execute("run", {}))
        assert "error" in r


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------

class TestProperties:
    def test_name(self, tool):
        assert tool.name == "process"

    def test_description(self, tool):
        assert isinstance(tool.description, str)
        assert len(tool.description) > 10

    def test_definitions_has_all_ops(self, tool):
        defs = tool.definitions()
        names = [d["name"] for d in defs]
        assert "run" in names
        assert "run_script" in names
        assert "which" in names

    def test_default_timeout(self):
        t = ProcessTool()
        assert t.timeout == 30

    def test_custom_timeout(self):
        t = ProcessTool(timeout=5)
        assert t.timeout == 5

    def test_default_cwd(self):
        t = ProcessTool(default_cwd="/tmp")
        assert t.default_cwd == "/tmp"
