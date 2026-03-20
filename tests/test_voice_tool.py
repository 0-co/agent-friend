"""Tests for VoiceTool — text-to-speech."""

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from agent_friend.tools.voice import VoiceTool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_http_response(body: bytes, status: int = 200):
    """Build a fake urllib response context manager."""
    mock_resp = MagicMock()
    mock_resp.read.return_value = body
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


# ---------------------------------------------------------------------------
# Construction / metadata
# ---------------------------------------------------------------------------

class TestVoiceToolInit(unittest.TestCase):
    def test_name(self):
        self.assertEqual(VoiceTool().name, "voice")

    def test_description_nonempty(self):
        self.assertTrue(len(VoiceTool().description) > 0)

    def test_definitions_count(self):
        self.assertEqual(len(VoiceTool().definitions()), 1)

    def test_definition_name_is_speak(self):
        self.assertEqual(VoiceTool().definitions()[0]["name"], "speak")

    def test_definition_has_input_schema(self):
        defn = VoiceTool().definitions()[0]
        self.assertIn("input_schema", defn)

    def test_definition_text_required(self):
        defn = VoiceTool().definitions()[0]
        self.assertIn("text", defn["input_schema"]["required"])

    def test_default_voice(self):
        tool = VoiceTool()
        self.assertEqual(tool.default_voice, "en-US-AriaNeural")

    def test_custom_default_voice(self):
        tool = VoiceTool(default_voice="en-US-BrianNeural")
        self.assertEqual(tool.default_voice, "en-US-BrianNeural")

    def test_save_dir_tilde_expanded(self):
        tool = VoiceTool(save_dir="~/.agent_friend/voice/")
        home = str(Path.home())
        self.assertTrue(tool.save_dir.startswith(home))
        self.assertNotIn("~", tool.save_dir)

    def test_tts_url_from_constructor(self):
        tool = VoiceTool(tts_url="http://localhost:8081")
        self.assertEqual(tool.tts_url, "http://localhost:8081")

    def test_tts_url_from_env(self):
        with patch.dict(os.environ, {"AGENT_FRIEND_TTS_URL": "http://tts.local:9000"}):
            tool = VoiceTool()
        self.assertEqual(tool.tts_url, "http://tts.local:9000")

    def test_tts_url_constructor_overrides_env(self):
        with patch.dict(os.environ, {"AGENT_FRIEND_TTS_URL": "http://env-server:8000"}):
            tool = VoiceTool(tts_url="http://explicit:8081")
        self.assertEqual(tool.tts_url, "http://explicit:8081")

    def test_tts_url_empty_when_not_configured(self):
        env = {k: v for k, v in os.environ.items() if k != "AGENT_FRIEND_TTS_URL"}
        with patch.dict(os.environ, env, clear=True):
            tool = VoiceTool()
        self.assertEqual(tool.tts_url, "")


# ---------------------------------------------------------------------------
# execute() — validation
# ---------------------------------------------------------------------------

class TestVoiceToolValidation(unittest.TestCase):
    def setUp(self):
        self.tool = VoiceTool()

    def test_unknown_tool_name_returns_error(self):
        result = self.tool.execute("shout", {"text": "hello"})
        self.assertIn("Unknown tool", result)

    def test_missing_text_returns_error(self):
        result = self.tool.execute("speak", {})
        self.assertIn("Error", result)
        self.assertIn("text", result)

    def test_empty_text_returns_error(self):
        result = self.tool.execute("speak", {"text": ""})
        self.assertIn("Error", result)

    def test_whitespace_only_text_returns_error(self):
        result = self.tool.execute("speak", {"text": "   "})
        self.assertIn("Error", result)


# ---------------------------------------------------------------------------
# HTTP backend
# ---------------------------------------------------------------------------

class TestVoiceToolHTTP(unittest.TestCase):
    def setUp(self):
        self.tool = VoiceTool(tts_url="http://localhost:8081")

    @patch("urllib.request.urlopen")
    def test_http_success_returns_saved_path(self, mock_urlopen):
        fake_mp3 = b"\xff\xfb\x90\x64" + b"\x00" * 100  # fake MP3 bytes
        mock_urlopen.return_value = _make_http_response(fake_mp3)

        with patch.object(self.tool, "_save_audio", return_value="/tmp/tts_123.mp3"):
            result = self.tool.execute("speak", {"text": "Hello world"})

        self.assertIn("neural TTS", result)
        self.assertIn("/tmp/tts_123.mp3", result)
        self.assertIn("11 chars", result)

    @patch("urllib.request.urlopen")
    def test_http_request_uses_correct_endpoint(self, mock_urlopen):
        mock_urlopen.return_value = _make_http_response(b"\xff\xfb" + b"\x00" * 50)

        with patch.object(self.tool, "_save_audio", return_value="/tmp/tts_x.mp3"):
            self.tool.execute("speak", {"text": "test"})

        call_args = mock_urlopen.call_args
        request_obj = call_args[0][0]
        self.assertEqual(request_obj.full_url, "http://localhost:8081/tts")

    @patch("urllib.request.urlopen")
    def test_http_request_sends_json_body(self, mock_urlopen):
        mock_urlopen.return_value = _make_http_response(b"\xff\xfb" + b"\x00" * 50)

        with patch.object(self.tool, "_save_audio", return_value="/tmp/tts_x.mp3"):
            self.tool.execute("speak", {"text": "hello", "voice": "en-US-BrianNeural"})

        request_obj = mock_urlopen.call_args[0][0]
        body = json.loads(request_obj.data.decode("utf-8"))
        self.assertEqual(body["text"], "hello")
        self.assertEqual(body["voice"], "en-US-BrianNeural")

    @patch("urllib.request.urlopen")
    def test_http_request_uses_default_voice(self, mock_urlopen):
        tool = VoiceTool(tts_url="http://localhost:8081", default_voice="en-US-EmmaNeural")
        mock_urlopen.return_value = _make_http_response(b"\xff\xfb" + b"\x00" * 50)

        with patch.object(tool, "_save_audio", return_value="/tmp/tts_x.mp3"):
            tool.execute("speak", {"text": "hello"})

        request_obj = mock_urlopen.call_args[0][0]
        body = json.loads(request_obj.data.decode("utf-8"))
        self.assertEqual(body["voice"], "en-US-EmmaNeural")

    @patch("urllib.request.urlopen")
    def test_http_connection_error_falls_back_to_system_tts(self, mock_urlopen):
        import urllib.error
        mock_urlopen.side_effect = urllib.error.URLError("Connection refused")

        # Patch system TTS to avoid actually running espeak
        with patch.object(self.tool, "_speak_via_system", return_value="Spoke via espeak: 5 chars") as mock_sys:
            result = self.tool.execute("speak", {"text": "hello"})

        mock_sys.assert_called_once_with("hello")
        self.assertIn("espeak", result)

    @patch("urllib.request.urlopen")
    def test_http_empty_response_falls_back_to_system_tts(self, mock_urlopen):
        mock_urlopen.return_value = _make_http_response(b"")

        with patch.object(self.tool, "_speak_via_system", return_value="Spoke via espeak: 4 chars"):
            result = self.tool.execute("speak", {"text": "test"})

        self.assertIn("espeak", result)


# ---------------------------------------------------------------------------
# save_dir creation and file saving
# ---------------------------------------------------------------------------

class TestVoiceToolSaveAudio(unittest.TestCase):
    def test_save_audio_creates_directory_and_file(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            save_dir = os.path.join(tmpdir, "tts_output")
            tool = VoiceTool(save_dir=save_dir)
            path = tool._save_audio(b"\xff\xfb" + b"\x00" * 20)

            self.assertIsNotNone(path)
            self.assertTrue(os.path.exists(path))
            self.assertTrue(path.startswith(save_dir))
            self.assertTrue(path.endswith(".mp3"))

    def test_save_audio_returns_none_on_bad_path(self):
        tool = VoiceTool(save_dir="/nonexistent_root/tts/")
        with patch("pathlib.Path.mkdir", side_effect=OSError("permission denied")):
            result = tool._save_audio(b"\xff\xfb" + b"\x00" * 20)
        self.assertIsNone(result)


# ---------------------------------------------------------------------------
# System TTS fallback
# ---------------------------------------------------------------------------

class TestVoiceToolSystemTTS(unittest.TestCase):
    def setUp(self):
        # No tts_url configured so HTTP backend is skipped.
        env = {k: v for k, v in os.environ.items() if k != "AGENT_FRIEND_TTS_URL"}
        with patch.dict(os.environ, env, clear=True):
            self.tool = VoiceTool()

    def test_no_tts_engine_returns_error_message(self):
        with patch("subprocess.run", side_effect=FileNotFoundError):
            result = self.tool.execute("speak", {"text": "hello"})
        self.assertIn("No TTS engine available", result)

    @patch("subprocess.run")
    def test_espeak_ng_success(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        result = self.tool._speak_via_system("hello")
        self.assertIn("espeak-ng", result)
        self.assertIn("5 chars", result)

    @patch("subprocess.run")
    def test_espeak_fallback_when_espeak_ng_missing(self, mock_run):
        def side_effect(args, **kwargs):
            if args[0] == "espeak-ng":
                raise FileNotFoundError
            return MagicMock(returncode=0)

        mock_run.side_effect = side_effect
        result = self.tool._speak_via_system("test")
        self.assertIn("espeak", result)

    @patch("subprocess.run")
    def test_festival_fallback_when_espeak_missing(self, mock_run):
        def side_effect(args, **kwargs):
            if args[0] in ("espeak-ng", "espeak"):
                raise FileNotFoundError
            return MagicMock(returncode=0)

        mock_run.side_effect = side_effect
        result = self.tool._speak_via_system("test")
        self.assertIn("festival", result)

    @patch("subprocess.run")
    def test_timeout_on_system_tts_returns_error(self, mock_run):
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="espeak-ng", timeout=30)
        result = self.tool._speak_via_system("hello")
        self.assertIn("No TTS engine available", result)

    def test_try_command_returns_none_when_not_found(self):
        with patch("subprocess.run", side_effect=FileNotFoundError):
            result = self.tool._try_command(["nonexistent_cmd", "text"], "nonexistent_cmd", 4)
        self.assertIsNone(result)

    def test_try_command_returns_none_on_nonzero_exit(self):
        with patch("subprocess.run", return_value=MagicMock(returncode=1)):
            result = self.tool._try_command(["espeak", "text"], "espeak", 4)
        self.assertIsNone(result)

    def test_try_festival_pipes_stdin(self):
        with patch("subprocess.run", return_value=MagicMock(returncode=0)) as mock_run:
            result = self.tool._try_festival("festival test")
        call_kwargs = mock_run.call_args[1]
        self.assertEqual(call_kwargs["input"], b"festival test")
        self.assertIn("festival", result)


# ---------------------------------------------------------------------------
# macOS / Windows platform paths (patched so they run on Linux CI)
# ---------------------------------------------------------------------------

class TestVoiceToolPlatformPaths(unittest.TestCase):
    def setUp(self):
        env = {k: v for k, v in os.environ.items() if k != "AGENT_FRIEND_TTS_URL"}
        with patch.dict(os.environ, env, clear=True):
            self.tool = VoiceTool()

    @patch("sys.platform", "darwin")
    @patch("subprocess.run")
    def test_macos_say_command_used(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        result = self.tool._speak_via_system("greetings")
        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][0]
        self.assertEqual(call_args[0], "say")
        self.assertIn("say", result)

    @patch("sys.platform", "darwin")
    @patch("subprocess.run", side_effect=FileNotFoundError)
    def test_macos_say_failure_returns_error(self, mock_run):
        result = self.tool._speak_via_system("hello")
        self.assertIn("Error", result)

    @patch("sys.platform", "win32")
    @patch("subprocess.run")
    def test_windows_powershell_used(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        result = self.tool._speak_via_system("hello windows")
        call_args = mock_run.call_args[0][0]
        self.assertEqual(call_args[0], "powershell")
        self.assertIn("powershell-sapi", result)

    @patch("sys.platform", "win32")
    @patch("subprocess.run", side_effect=FileNotFoundError)
    def test_windows_powershell_failure_returns_error(self, mock_run):
        result = self.tool._speak_via_system("hello")
        self.assertIn("Error", result)


# ---------------------------------------------------------------------------
# Integration with friend.py tool registry
# ---------------------------------------------------------------------------

class TestVoiceToolRegistration(unittest.TestCase):
    def test_voice_in_tool_name_map(self):
        from agent_friend.friend import _TOOL_NAME_MAP
        self.assertIn("voice", _TOOL_NAME_MAP)

    def test_voice_importable_from_top_level(self):
        from agent_friend import VoiceTool as VT
        self.assertIs(VT, VoiceTool)

    def test_voice_importable_from_tools_init(self):
        from agent_friend.tools import VoiceTool as VT2
        self.assertIs(VT2, VoiceTool)


if __name__ == "__main__":
    unittest.main()
