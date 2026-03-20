"""voice.py — Text-to-speech tool for agent-friend (no required dependencies)."""

import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import BaseTool


class VoiceTool(BaseTool):
    """Speak text aloud using a neural TTS server or system TTS fallback.

    Attempts the HTTP TTS server first (if a URL is configured), then falls
    back to whatever system TTS engine is available (espeak-ng, espeak,
    festival on Linux; say on macOS; PowerShell SpeechSynthesizer on Windows).

    Parameters
    ----------
    tts_url:       Base URL of the HTTP TTS server (e.g. "http://localhost:8081").
                   Falls back to the AGENT_FRIEND_TTS_URL environment variable.
    save_dir:      Directory to save MP3 files from the HTTP backend.
                   Defaults to ``~/.agent_friend/voice/``.
    default_voice: Voice name sent to the HTTP TTS server.
                   Defaults to ``"en-US-AriaNeural"``.
    """

    def __init__(
        self,
        tts_url: Optional[str] = None,
        save_dir: str = "~/.agent_friend/voice/",
        default_voice: str = "en-US-AriaNeural",
    ) -> None:
        self.tts_url = tts_url or os.environ.get("AGENT_FRIEND_TTS_URL", "")
        self.save_dir = str(Path(save_dir).expanduser())
        self.default_voice = default_voice

    # ------------------------------------------------------------------
    # BaseTool interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "voice"

    @property
    def description(self) -> str:
        return (
            "Speak text aloud using neural TTS (HTTP server) or system TTS. "
            "No API key required for system TTS."
        )

    def definitions(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "speak",
                "description": (
                    "Convert text to speech and play it aloud. "
                    "Uses a neural TTS server if configured, otherwise falls back "
                    "to the system TTS engine (espeak, festival, say, etc.)."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "The text to speak aloud.",
                        },
                        "voice": {
                            "type": "string",
                            "description": (
                                "Voice name for the HTTP TTS server "
                                "(e.g. 'en-US-AriaNeural'). Ignored for system TTS."
                            ),
                        },
                    },
                    "required": ["text"],
                },
            }
        ]

    def execute(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        if tool_name != "speak":
            return f"Unknown tool: {tool_name}"

        text = arguments.get("text", "").strip()
        if not text:
            return "Error: text is required and cannot be empty."

        voice = arguments.get("voice") or self.default_voice

        # Try the HTTP TTS server first if a URL is configured.
        if self.tts_url:
            result = self._speak_via_http(text, voice)
            if result is not None:
                return result

        # Fall back to system TTS.
        return self._speak_via_system(text)

    # ------------------------------------------------------------------
    # HTTP backend
    # ------------------------------------------------------------------

    def _speak_via_http(self, text: str, voice: str) -> Optional[str]:
        """POST to the TTS server and save the returned MP3.

        Returns a success string on success, or None if the request fails
        (so the caller can fall back to system TTS).
        """
        endpoint = self.tts_url.rstrip("/") + "/tts"
        payload = json.dumps({"text": text, "voice": voice}).encode("utf-8")
        req = urllib.request.Request(
            endpoint,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                audio_bytes = resp.read()
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError):
            # Server unreachable — fall through to system TTS.
            return None

        if not audio_bytes:
            return None

        save_path = self._save_audio(audio_bytes)
        if save_path is None:
            return None

        return (
            f"Spoke via neural TTS: saved to {save_path} ({len(text)} chars)"
        )

    def _save_audio(self, audio_bytes: bytes) -> Optional[str]:
        """Write MP3 bytes to a timestamped file in save_dir.

        Returns the file path on success, or None on any I/O error.
        """
        try:
            Path(self.save_dir).mkdir(parents=True, exist_ok=True)
            timestamp = int(time.time())
            filename = f"tts_{timestamp}.mp3"
            save_path = os.path.join(self.save_dir, filename)
            with open(save_path, "wb") as audio_file:
                audio_file.write(audio_bytes)
            return save_path
        except OSError:
            return None

    # ------------------------------------------------------------------
    # System TTS backend
    # ------------------------------------------------------------------

    def _speak_via_system(self, text: str) -> str:
        """Try each system TTS engine in order.

        Returns a success string or a descriptive error message.
        """
        platform = sys.platform

        if platform == "darwin":
            return self._try_macos_say(text)

        if platform.startswith("win"):
            return self._try_windows_sapi(text)

        # Linux and everything else — try espeak-ng, espeak, then festival.
        result = self._try_command(["espeak-ng", "-a", "200", text], "espeak-ng", len(text))
        if result is not None:
            return result

        result = self._try_command(["espeak", "-a", "200", text], "espeak", len(text))
        if result is not None:
            return result

        result = self._try_festival(text)
        if result is not None:
            return result

        return (
            "No TTS engine available. "
            "Install espeak-ng, espeak, or festival; or set AGENT_FRIEND_TTS_URL."
        )

    def _try_command(
        self, args: List[str], engine_name: str, char_count: int
    ) -> Optional[str]:
        """Run a subprocess TTS command.

        Returns a success string on exit code 0, or None if the command is not
        found or fails.
        """
        try:
            result = subprocess.run(
                args,
                capture_output=True,
                timeout=30,
            )
            if result.returncode == 0:
                return f"Spoke via {engine_name}: {char_count} chars"
            return None
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            return None

    def _try_festival(self, text: str) -> Optional[str]:
        """Pipe text into festival --tts via stdin."""
        try:
            result = subprocess.run(
                ["festival", "--tts"],
                input=text.encode("utf-8"),
                capture_output=True,
                timeout=30,
            )
            if result.returncode == 0:
                return f"Spoke via festival: {len(text)} chars"
            return None
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            return None

    def _try_macos_say(self, text: str) -> str:
        """Use macOS built-in say command."""
        args = ["say", text]
        result = self._try_command(args, "say", len(text))
        if result is not None:
            return result
        return "Error: macOS 'say' command failed or not available."

    def _try_windows_sapi(self, text: str) -> str:
        """Use PowerShell System.Speech.Synthesis.SpeechSynthesizer on Windows."""
        script = (
            "Add-Type -AssemblyName System.Speech; "
            "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            f'$s.Speak("{text.replace(chr(34), chr(39))}")'
        )
        result = self._try_command(
            ["powershell", "-Command", script], "powershell-sapi", len(text)
        )
        if result is not None:
            return result
        return "Error: PowerShell SpeechSynthesizer failed or not available."
