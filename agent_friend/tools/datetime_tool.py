"""datetime_tool.py — DateTimeTool for agent-friend (stdlib only).

Agents can perform date and time operations without CodeTool: get the
current time, parse date strings, format datetimes, compute differences,
add/subtract durations, and convert between timezones.

Uses only Python stdlib: ``datetime``, ``zoneinfo`` (Python 3.9+).

Usage::

    tool = DateTimeTool()
    tool.now()                                    # "2026-03-12T02:53:00+00:00"
    tool.now(timezone="America/New_York")          # "2026-03-11T22:53:00-04:00"
    tool.parse("March 12, 2026 15:30")            # ISO 8601 string
    tool.add_duration("2026-03-12T00:00:00", days=7)  # one week later
    tool.diff("2026-03-12", "2026-04-01", unit="days") # "20"
    tool.convert_timezone("2026-03-12T10:00:00", to_tz="Asia/Tokyo")
"""

import json
from datetime import datetime, timedelta, timezone, date
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .base import BaseTool


_PARSE_FORMATS = [
    "%Y-%m-%dT%H:%M:%S%z",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%d",
    "%m/%d/%Y %H:%M:%S",
    "%m/%d/%Y %H:%M",
    "%m/%d/%Y",
    "%d/%m/%Y",
    "%B %d, %Y %H:%M",
    "%B %d, %Y",
    "%b %d, %Y",
    "%d %B %Y",
    "%d %b %Y",
]


def _resolve_tz(tz_name: str) -> Any:
    """Return a timezone object from a name string."""
    if tz_name.upper() in ("UTC", "Z"):
        return timezone.utc
    try:
        return ZoneInfo(tz_name)
    except ZoneInfoNotFoundError:
        raise ValueError(f"Unknown timezone: {tz_name!r}. Use IANA names like 'America/New_York' or 'UTC'.")


def _parse_dt(text: str, default_tz: Any = timezone.utc) -> datetime:
    """Parse a datetime string using multiple formats.  Returns an
    aware datetime.  If no timezone info is embedded, *default_tz* is used."""
    text = text.strip()
    for fmt in _PARSE_FORMATS:
        try:
            dt = datetime.strptime(text, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=default_tz)
            return dt
        except ValueError:
            continue
    raise ValueError(
        f"Unable to parse date/time: {text!r}. "
        "Accepted formats: ISO 8601 (2026-03-12T15:30:00), "
        "YYYY-MM-DD, MM/DD/YYYY, 'March 12, 2026', etc."
    )


class DateTimeTool(BaseTool):
    """Date and time operations for AI agents.

    Provides current time, date parsing, formatting, arithmetic, timezone
    conversion, and duration calculations — all without external libraries.

    Works with ISO 8601 datetimes and common human-readable formats.
    """

    name = "datetime"
    description = (
        "Date and time operations. Get current time, parse date strings, "
        "format datetimes, add/subtract durations, compute differences, "
        "and convert between timezones. Uses IANA timezone names "
        "(e.g. 'America/New_York', 'Europe/London', 'Asia/Tokyo')."
    )

    # ── public Python API ─────────────────────────────────────────────────

    def now(self, timezone: str = "UTC") -> str:
        """Return the current date and time as an ISO 8601 string.

        Parameters
        ----------
        timezone: IANA timezone name, default ``"UTC"``.
        """
        tz = _resolve_tz(timezone)
        return datetime.now(tz=tz).isoformat()

    def parse(self, text: str, default_timezone: str = "UTC") -> str:
        """Parse a date/time string and return it in ISO 8601 format.

        Accepts ISO 8601, ``YYYY-MM-DD``, ``MM/DD/YYYY``, natural-language
        month names (``"March 12, 2026"``), and more.

        Parameters
        ----------
        text: Date/time string to parse.
        default_timezone: IANA timezone to assume when *text* has no tz info.
        """
        tz = _resolve_tz(default_timezone)
        dt = _parse_dt(text, default_tz=tz)
        return dt.isoformat()

    def format_dt(self, dt_str: str, fmt: str = "%Y-%m-%d %H:%M:%S", timezone: str = "UTC") -> str:
        """Format a datetime string according to *fmt*.

        Parameters
        ----------
        dt_str: ISO 8601 or other supported date string.
        fmt: strftime format string, e.g. ``"%B %d, %Y"`` → ``"March 12, 2026"``.
        timezone: IANA timezone to localise to before formatting.
        """
        tz = _resolve_tz(timezone)
        dt = _parse_dt(dt_str)
        dt_local = dt.astimezone(tz)
        return dt_local.strftime(fmt)

    def diff(self, a: str, b: str, unit: str = "seconds") -> str:
        """Compute the absolute difference between two datetime strings.

        Parameters
        ----------
        a, b: Datetime strings (any supported format).
        unit: One of ``"seconds"``, ``"minutes"``, ``"hours"``, ``"days"``.
              Returns a float rounded to two decimal places.
        """
        dt_a = _parse_dt(a)
        dt_b = _parse_dt(b)
        delta = abs(dt_b - dt_a)
        total_seconds = delta.total_seconds()

        if unit == "seconds":
            result = total_seconds
        elif unit == "minutes":
            result = total_seconds / 60
        elif unit == "hours":
            result = total_seconds / 3600
        elif unit == "days":
            result = total_seconds / 86400
        else:
            raise ValueError(f"Unknown unit: {unit!r}. Use 'seconds', 'minutes', 'hours', or 'days'.")

        return str(round(result, 2))

    def add_duration(
        self,
        dt_str: str,
        days: int = 0,
        hours: int = 0,
        minutes: int = 0,
        seconds: int = 0,
    ) -> str:
        """Add (or subtract with negative values) a duration to a datetime.

        Parameters
        ----------
        dt_str: Starting datetime string.
        days, hours, minutes, seconds: Duration to add. Negative values subtract.

        Returns ISO 8601 string.
        """
        dt = _parse_dt(dt_str)
        delta = timedelta(days=days, hours=hours, minutes=minutes, seconds=seconds)
        return (dt + delta).isoformat()

    def convert_timezone(self, dt_str: str, to_tz: str, from_tz: str = "UTC") -> str:
        """Convert a datetime from one timezone to another.

        Parameters
        ----------
        dt_str: Datetime string. If it has embedded timezone info that is used;
                otherwise *from_tz* is assumed.
        to_tz: Target IANA timezone name.
        from_tz: Source timezone if *dt_str* has no timezone info.
        """
        src_tz = _resolve_tz(from_tz)
        tgt_tz = _resolve_tz(to_tz)
        dt = _parse_dt(dt_str, default_tz=src_tz)
        return dt.astimezone(tgt_tz).isoformat()

    def to_timestamp(self, dt_str: str) -> str:
        """Convert a datetime string to a Unix timestamp (seconds since epoch).

        Returns a string representation of the integer timestamp.
        """
        dt = _parse_dt(dt_str)
        return str(int(dt.timestamp()))

    def from_timestamp(self, timestamp: str, timezone: str = "UTC") -> str:
        """Convert a Unix timestamp to an ISO 8601 datetime string.

        Parameters
        ----------
        timestamp: Unix timestamp as a string (int or float).
        timezone: IANA timezone for the output.
        """
        tz = _resolve_tz(timezone)
        ts = float(timestamp)
        dt = datetime.fromtimestamp(ts, tz=tz)
        return dt.isoformat()

    # ── LLM tool dispatch ─────────────────────────────────────────────────

    def definitions(self) -> List[Dict]:
        return [
            {
                "name": "now",
                "description": "Return the current date and time as ISO 8601.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "timezone": {
                            "type": "string",
                            "description": "IANA timezone name, e.g. 'UTC', 'America/New_York'. Default: UTC.",
                        }
                    },
                },
            },
            {
                "name": "parse",
                "description": "Parse a date/time string and return ISO 8601.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "Date/time string to parse."},
                        "default_timezone": {
                            "type": "string",
                            "description": "IANA timezone to assume if text has none. Default: UTC.",
                        },
                    },
                    "required": ["text"],
                },
            },
            {
                "name": "format_dt",
                "description": "Format a datetime string using a strftime format.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "dt_str": {"type": "string", "description": "Datetime string to format."},
                        "fmt": {
                            "type": "string",
                            "description": "strftime format, e.g. '%B %d, %Y'. Default: '%Y-%m-%d %H:%M:%S'.",
                        },
                        "timezone": {
                            "type": "string",
                            "description": "IANA timezone to convert to before formatting. Default: UTC.",
                        },
                    },
                    "required": ["dt_str"],
                },
            },
            {
                "name": "diff",
                "description": "Compute the absolute difference between two datetime strings.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "string", "description": "First datetime string."},
                        "b": {"type": "string", "description": "Second datetime string."},
                        "unit": {
                            "type": "string",
                            "enum": ["seconds", "minutes", "hours", "days"],
                            "description": "Unit for the result. Default: 'seconds'.",
                        },
                    },
                    "required": ["a", "b"],
                },
            },
            {
                "name": "add_duration",
                "description": "Add (or subtract) a duration to a datetime.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "dt_str": {"type": "string", "description": "Starting datetime string."},
                        "days": {"type": "integer", "description": "Days to add (negative to subtract)."},
                        "hours": {"type": "integer", "description": "Hours to add."},
                        "minutes": {"type": "integer", "description": "Minutes to add."},
                        "seconds": {"type": "integer", "description": "Seconds to add."},
                    },
                    "required": ["dt_str"],
                },
            },
            {
                "name": "convert_timezone",
                "description": "Convert a datetime string from one timezone to another.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "dt_str": {"type": "string", "description": "Datetime string to convert."},
                        "to_tz": {"type": "string", "description": "Target IANA timezone name."},
                        "from_tz": {
                            "type": "string",
                            "description": "Source timezone if dt_str has none. Default: UTC.",
                        },
                    },
                    "required": ["dt_str", "to_tz"],
                },
            },
            {
                "name": "to_timestamp",
                "description": "Convert a datetime string to a Unix timestamp (seconds since epoch).",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "dt_str": {"type": "string", "description": "Datetime string to convert."},
                    },
                    "required": ["dt_str"],
                },
            },
            {
                "name": "from_timestamp",
                "description": "Convert a Unix timestamp to an ISO 8601 datetime string.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "timestamp": {"type": "string", "description": "Unix timestamp as a string."},
                        "timezone": {
                            "type": "string",
                            "description": "IANA timezone for the output. Default: UTC.",
                        },
                    },
                    "required": ["timestamp"],
                },
            },
        ]

    def execute(self, tool_name: str, tool_input: Dict) -> str:
        try:
            if tool_name == "now":
                return self.now(timezone=tool_input.get("timezone", "UTC"))
            elif tool_name == "parse":
                return self.parse(
                    text=tool_input["text"],
                    default_timezone=tool_input.get("default_timezone", "UTC"),
                )
            elif tool_name == "format_dt":
                return self.format_dt(
                    dt_str=tool_input["dt_str"],
                    fmt=tool_input.get("fmt", "%Y-%m-%d %H:%M:%S"),
                    timezone=tool_input.get("timezone", "UTC"),
                )
            elif tool_name == "diff":
                return self.diff(
                    a=tool_input["a"],
                    b=tool_input["b"],
                    unit=tool_input.get("unit", "seconds"),
                )
            elif tool_name == "add_duration":
                return self.add_duration(
                    dt_str=tool_input["dt_str"],
                    days=tool_input.get("days", 0),
                    hours=tool_input.get("hours", 0),
                    minutes=tool_input.get("minutes", 0),
                    seconds=tool_input.get("seconds", 0),
                )
            elif tool_name == "convert_timezone":
                return self.convert_timezone(
                    dt_str=tool_input["dt_str"],
                    to_tz=tool_input["to_tz"],
                    from_tz=tool_input.get("from_tz", "UTC"),
                )
            elif tool_name == "to_timestamp":
                return self.to_timestamp(dt_str=tool_input["dt_str"])
            elif tool_name == "from_timestamp":
                return self.from_timestamp(
                    timestamp=tool_input["timestamp"],
                    timezone=tool_input.get("timezone", "UTC"),
                )
            else:
                return f"Unknown tool: {tool_name}"
        except (KeyError, ValueError, TypeError, ZoneInfoNotFoundError) as e:
            return f"Error: {e}"
