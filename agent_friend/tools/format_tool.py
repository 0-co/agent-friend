"""format_tool.py — FormatTool for agent-friend (stdlib only).

Human-readable formatting for numbers, file sizes, durations, percentages,
and currency.  Agents deal with raw numbers from APIs constantly; FormatTool
makes them readable without CodeTool.

Features:
* format_bytes   — 1234567 → "1.2 MB"
* format_number  — 1234567.89 → "1,234,567.89"
* format_duration — 3661 → "1h 1m 1s"
* format_percent — 0.8734 → "87.3%"
* format_currency — 1234.5, "USD" → "$1,234.50"
* format_ordinal — 1 → "1st", 3 → "3rd", 11 → "11th"
* format_plural  — pluralize a noun based on a count
* format_truncate — truncate long strings with ellipsis
* format_pad     — left/right/center-pad a string
* format_table   — render a list of dicts as a plain-text table

Usage::

    tool = FormatTool()

    tool.format_bytes(1_234_567)     # "1.2 MB"
    tool.format_duration(3_661)      # "1h 1m 1s"
    tool.format_number(1_234_567.89) # "1,234,567.89"
    tool.format_percent(0.8734, 1)   # "87.3%"
"""

import json
import math
from typing import Any, Dict, List, Optional

from .base import BaseTool


_BYTE_UNITS = ["B", "KB", "MB", "GB", "TB", "PB", "EB"]

_CURRENCY_SYMBOLS: Dict[str, str] = {
    "USD": "$", "EUR": "€", "GBP": "£", "JPY": "¥",
    "CAD": "CA$", "AUD": "A$", "CHF": "CHF ", "CNY": "¥",
    "INR": "₹", "KRW": "₩", "BRL": "R$", "MXN": "MX$",
}


def _ordinal_suffix(n: int) -> str:
    n = abs(n)
    if 11 <= (n % 100) <= 13:
        return "th"
    remainder = n % 10
    if remainder == 1:
        return "st"
    if remainder == 2:
        return "nd"
    if remainder == 3:
        return "rd"
    return "th"


class FormatTool(BaseTool):
    """Human-readable formatting: bytes, durations, numbers, percents, tables.

    Pairs naturally with MetricsTool, HTTPTool, and DatabaseTool — anywhere
    you get raw numbers that need to be displayed to a user or included in
    a report.  Zero dependencies.
    """

    # ── public API ────────────────────────────────────────────────────

    def format_bytes(self, value: float, decimals: int = 1, binary: bool = False) -> str:
        """Format a byte count as a human-readable string.

        Parameters
        ----------
        value:
            Number of bytes (int or float).
        decimals:
            Decimal places (default 1).
        binary:
            If true use 1024-based units (KiB, MiB, …), otherwise 1000-based (KB, MB, …).

        Returns a plain string like ``"1.2 MB"`` or ``"4.0 KiB"``.
        """
        try:
            v = float(value)
        except (TypeError, ValueError):
            return json.dumps({"error": f"Invalid value: {value!r}"})

        if v < 0:
            sign = "-"
            v = -v
        else:
            sign = ""

        base = 1024 if binary else 1000
        units = (
            ["B", "KiB", "MiB", "GiB", "TiB", "PiB", "EiB"]
            if binary
            else _BYTE_UNITS
        )

        if v < base:
            return f"{sign}{v:.{decimals}f} B" if decimals else f"{sign}{int(v)} B"

        exp = int(math.log(v, base))
        exp = min(exp, len(units) - 1)
        val = v / base ** exp
        return f"{sign}{val:.{decimals}f} {units[exp]}"

    def format_duration(self, seconds: float, verbose: bool = False) -> str:
        """Format a duration in seconds as a human-readable string.

        Examples:

        * ``format_duration(90)`` → ``"1m 30s"``
        * ``format_duration(3661)`` → ``"1h 1m 1s"``
        * ``format_duration(86400)`` → ``"1d 0h 0m 0s"``
        * ``format_duration(0.45)`` → ``"450ms"``
        * ``format_duration(90, verbose=True)`` → ``"1 minute 30 seconds"``
        """
        try:
            secs = float(seconds)
        except (TypeError, ValueError):
            return json.dumps({"error": f"Invalid value: {seconds!r}"})

        if secs < 0:
            return "0s"

        if secs < 1:
            ms = int(round(secs * 1000))
            if verbose:
                return f"{ms} millisecond{'s' if ms != 1 else ''}"
            return f"{ms}ms"

        total = int(secs)
        days, rem = divmod(total, 86400)
        hours, rem = divmod(rem, 3600)
        minutes, secs_i = divmod(rem, 60)

        if verbose:
            parts = []
            if days:
                parts.append(f"{days} day{'s' if days != 1 else ''}")
            if hours:
                parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
            if minutes:
                parts.append(f"{minutes} minute{'s' if minutes != 1 else ''}")
            if secs_i or not parts:
                parts.append(f"{secs_i} second{'s' if secs_i != 1 else ''}")
            return " ".join(parts)

        if days:
            return f"{days}d {hours}h {minutes}m {secs_i}s"
        if hours:
            return f"{hours}h {minutes}m {secs_i}s"
        if minutes:
            return f"{minutes}m {secs_i}s"
        return f"{secs_i}s"

    def format_number(
        self,
        value: float,
        decimals: int = 2,
        thousands_sep: str = ",",
        decimal_sep: str = ".",
    ) -> str:
        """Format a number with thousands separators.

        Examples:

        * ``format_number(1234567.89)`` → ``"1,234,567.89"``
        * ``format_number(1234567, decimals=0)`` → ``"1,234,567"``
        """
        try:
            v = float(value)
        except (TypeError, ValueError):
            return json.dumps({"error": f"Invalid value: {value!r}"})

        # Build integer and fractional parts
        negative = v < 0
        v = abs(v)
        rounded = round(v, decimals)
        int_part = int(rounded)
        frac = rounded - int_part

        # Thousands-separate the integer part
        int_str = str(int_part)
        groups = []
        while len(int_str) > 3:
            groups.insert(0, int_str[-3:])
            int_str = int_str[:-3]
        groups.insert(0, int_str)
        formatted_int = thousands_sep.join(groups)

        if decimals > 0:
            frac_str = f"{frac:.{decimals}f}"[1:]  # ".XY"
            result = formatted_int + decimal_sep + frac_str[1:]  # strip leading dot
        else:
            result = formatted_int

        return ("-" if negative else "") + result

    def format_percent(self, value: float, decimals: int = 1, include_sign: bool = False) -> str:
        """Format a ratio (0–1) or raw percentage as a percent string.

        If *value* is between -1 and 1 (exclusive of -1 and 1 edges check
        skipped — just multiplies by 100 if |value| ≤ 1 and decimals make sense),
        it is assumed to be a ratio and multiplied by 100.  Otherwise it is
        treated as already a percentage.

        Examples:

        * ``format_percent(0.8734)`` → ``"87.3%"``
        * ``format_percent(87.34)`` → ``"87.3%"``  (>1, treated as already %)
        * ``format_percent(0.05, include_sign=True)`` → ``"+5.0%"``
        """
        try:
            v = float(value)
        except (TypeError, ValueError):
            return json.dumps({"error": f"Invalid value: {value!r}"})

        # Heuristic: if -1 < v <= 1, treat as ratio
        if -1 < v <= 1:
            v *= 100

        sign = "+" if include_sign and v > 0 else ""
        return f"{sign}{v:.{decimals}f}%"

    def format_currency(
        self,
        value: float,
        currency: str = "USD",
        decimals: int = 2,
        thousands_sep: str = ",",
    ) -> str:
        """Format a monetary value.

        Examples:

        * ``format_currency(1234.5)`` → ``"$1,234.50"``
        * ``format_currency(1234.5, 'EUR')`` → ``"€1,234.50"``
        * ``format_currency(1234.5, 'JPY', decimals=0)`` → ``"¥1,235"``
        """
        try:
            v = float(value)
        except (TypeError, ValueError):
            return json.dumps({"error": f"Invalid value: {value!r}"})

        symbol = _CURRENCY_SYMBOLS.get(currency.upper(), currency + " ")
        negative = v < 0
        formatted = self.format_number(abs(v), decimals=decimals, thousands_sep=thousands_sep)
        result = f"{symbol}{formatted}"
        return f"-{result}" if negative else result

    def format_ordinal(self, n: int) -> str:
        """Return the ordinal string for an integer.

        Examples: 1 → "1st", 2 → "2nd", 3 → "3rd", 11 → "11th", 21 → "21st".
        """
        try:
            n = int(n)
        except (TypeError, ValueError):
            return json.dumps({"error": f"Invalid value: {n!r}"})
        return f"{n}{_ordinal_suffix(n)}"

    def format_plural(
        self,
        count: float,
        singular: str,
        plural: Optional[str] = None,
        include_count: bool = True,
    ) -> str:
        """Return ``"N word"`` or ``"N words"`` based on *count*.

        If *plural* is omitted, appends ``"s"`` to *singular*.

        Examples:

        * ``format_plural(1, "item")`` → ``"1 item"``
        * ``format_plural(3, "item")`` → ``"3 items"``
        * ``format_plural(1, "mouse", "mice")`` → ``"1 mouse"``
        """
        try:
            count = float(count)
        except (TypeError, ValueError):
            return json.dumps({"error": f"Invalid count: {count!r}"})

        word = singular if count == 1 else (plural or singular + "s")
        if include_count:
            cnt = int(count) if count == int(count) else count
            return f"{cnt} {word}"
        return word

    def format_truncate(self, text: str, max_length: int = 80, suffix: str = "…") -> str:
        """Truncate *text* to *max_length* characters, appending *suffix*.

        If the text is already short enough, return it unchanged.
        """
        if not isinstance(text, str):
            return json.dumps({"error": "text must be a string"})
        if len(text) <= max_length:
            return text
        cut = max_length - len(suffix)
        if cut < 0:
            return suffix[:max_length]
        return text[:cut] + suffix

    def format_pad(
        self,
        text: str,
        width: int,
        align: str = "left",
        fill: str = " ",
    ) -> str:
        """Pad *text* to *width* characters.

        *align* is ``"left"``, ``"right"``, or ``"center"``.
        """
        if not isinstance(text, str):
            return json.dumps({"error": "text must be a string"})
        if align == "left":
            return text.ljust(width, fill[0] if fill else " ")
        if align == "right":
            return text.rjust(width, fill[0] if fill else " ")
        if align == "center":
            return text.center(width, fill[0] if fill else " ")
        return json.dumps({"error": f"Unknown align '{align}'. Use left/right/center."})

    def format_table(
        self,
        data: str,
        columns: Optional[List[str]] = None,
        max_col_width: int = 30,
    ) -> str:
        """Render a JSON array of dicts as a plain-text table.

        *columns* specifies column order; defaults to keys of the first row.
        *max_col_width* truncates long values.

        Returns the table as a plain string (not JSON).
        """
        try:
            rows = json.loads(data)
        except json.JSONDecodeError as exc:
            return f"Error: Invalid JSON: {exc}"

        if not isinstance(rows, list) or not rows:
            return "(empty)"

        if columns is None:
            cols = list(rows[0].keys()) if isinstance(rows[0], dict) else []
        else:
            cols = columns

        if not cols:
            return str(rows)

        # Calculate column widths
        widths = {c: len(str(c)) for c in cols}
        for row in rows:
            if isinstance(row, dict):
                for c in cols:
                    val = str(row.get(c, ""))
                    if len(val) > max_col_width:
                        val = val[:max_col_width - 1] + "…"
                    widths[c] = max(widths[c], len(val))

        # Build table
        sep = "+" + "+".join("-" * (w + 2) for w in widths.values()) + "+"
        header = "|" + "|".join(f" {c.center(widths[c])} " for c in cols) + "|"
        lines = [sep, header, sep]

        for row in rows:
            if isinstance(row, dict):
                cells = []
                for c in cols:
                    val = str(row.get(c, ""))
                    if len(val) > max_col_width:
                        val = val[:max_col_width - 1] + "…"
                    cells.append(f" {val.ljust(widths[c])} ")
                lines.append("|" + "|".join(cells) + "|")
            else:
                max_w = sum(widths.values())
                lines.append(f"| {str(row)[:max_w]} |")

        lines.append(sep)
        return "\n".join(lines)

    # ── BaseTool interface ────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "format"

    @property
    def description(self) -> str:
        return (
            "Human-readable formatting: bytes, durations, numbers, percents, currency, "
            "ordinals, plurals, truncation, padding, and plain-text tables. "
            "Pairs with MetricsTool and HTTPTool. Zero deps."
        )

    def definitions(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "format_bytes",
                "description": "Format a byte count: 1234567 → '1.2 MB'. binary=true for KiB/MiB.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "value": {"type": "number"},
                        "decimals": {"type": "integer", "description": "Decimal places (default 1)"},
                        "binary": {"type": "boolean", "description": "Use 1024-based units (KiB/MiB/GiB)"},
                    },
                    "required": ["value"],
                },
            },
            {
                "name": "format_duration",
                "description": "Format seconds as a human string: 3661 → '1h 1m 1s'. verbose=true for full words.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "seconds": {"type": "number"},
                        "verbose": {"type": "boolean"},
                    },
                    "required": ["seconds"],
                },
            },
            {
                "name": "format_number",
                "description": "Format with thousands separators: 1234567.89 → '1,234,567.89'.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "value": {"type": "number"},
                        "decimals": {"type": "integer"},
                        "thousands_sep": {"type": "string"},
                        "decimal_sep": {"type": "string"},
                    },
                    "required": ["value"],
                },
            },
            {
                "name": "format_percent",
                "description": "Format as percent: 0.87 → '87.0%'. Ratios (|v|<1) multiplied by 100.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "value": {"type": "number"},
                        "decimals": {"type": "integer"},
                        "include_sign": {"type": "boolean"},
                    },
                    "required": ["value"],
                },
            },
            {
                "name": "format_currency",
                "description": "Format as currency: 1234.5 → '$1,234.50'. currency: USD/EUR/GBP/JPY/...",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "value": {"type": "number"},
                        "currency": {"type": "string", "description": "ISO 4217 code (USD/EUR/GBP/JPY/...)"},
                        "decimals": {"type": "integer"},
                    },
                    "required": ["value"],
                },
            },
            {
                "name": "format_ordinal",
                "description": "1 → '1st', 2 → '2nd', 3 → '3rd', 11 → '11th', 21 → '21st'.",
                "input_schema": {
                    "type": "object",
                    "properties": {"n": {"type": "integer"}},
                    "required": ["n"],
                },
            },
            {
                "name": "format_plural",
                "description": "Pluralize: format_plural(3, 'item') → '3 items'. Optional custom plural form.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "count": {"type": "number"},
                        "singular": {"type": "string"},
                        "plural": {"type": "string"},
                        "include_count": {"type": "boolean"},
                    },
                    "required": ["count", "singular"],
                },
            },
            {
                "name": "format_truncate",
                "description": "Truncate text to max_length, appending suffix (default '…').",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "max_length": {"type": "integer"},
                        "suffix": {"type": "string"},
                    },
                    "required": ["text"],
                },
            },
            {
                "name": "format_pad",
                "description": "Pad text to width. align: left/right/center.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "width": {"type": "integer"},
                        "align": {"type": "string"},
                        "fill": {"type": "string"},
                    },
                    "required": ["text", "width"],
                },
            },
            {
                "name": "format_table",
                "description": "Render a JSON array of dicts as a plain-text table.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "data": {"type": "string", "description": "JSON array of dicts"},
                        "columns": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Column names in order (default: keys of first row)",
                        },
                        "max_col_width": {"type": "integer"},
                    },
                    "required": ["data"],
                },
            },
        ]

    def execute(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        if tool_name == "format_bytes":
            return self.format_bytes(**arguments)
        if tool_name == "format_duration":
            return self.format_duration(**arguments)
        if tool_name == "format_number":
            return self.format_number(**arguments)
        if tool_name == "format_percent":
            return self.format_percent(**arguments)
        if tool_name == "format_currency":
            return self.format_currency(**arguments)
        if tool_name == "format_ordinal":
            return self.format_ordinal(**arguments)
        if tool_name == "format_plural":
            return self.format_plural(**arguments)
        if tool_name == "format_truncate":
            return self.format_truncate(**arguments)
        if tool_name == "format_pad":
            return self.format_pad(**arguments)
        if tool_name == "format_table":
            return self.format_table(**arguments)
        return json.dumps({"error": f"Unknown tool: {tool_name}"})
