"""table.py — TableTool for agent-friend (CSV/TSV, no required dependencies).

Agents can read, filter, aggregate, and write tabular data without pandas.
Uses stdlib only: csv, json, statistics, os, pathlib.

Usage::

    tool = TableTool()
    rows = tool.read("data.csv")
    filtered = tool.filter_rows("data.csv", "age", "gt", "30")
    result = tool.aggregate("data.csv", "salary", "avg")
    tool.write("out.csv", rows)
    tool.append_row("data.csv", {"name": "Alice", "age": "25"})
"""

import csv
import json
import os
import statistics
from pathlib import Path
from typing import Any, Dict, List

from .base import BaseTool


_DEFAULT_BASE_DIR = "~/.agent_friend/tables"


class TableTool(BaseTool):
    """CSV/TSV reader and writer with structured query operations.

    Lets agents read, filter, aggregate, and write tabular data
    without writing pandas code. Uses stdlib only (csv, statistics).

    Parameters
    ----------
    base_dir:
        Default directory for relative paths.
        Defaults to ``~/.agent_friend/tables``.
    """

    def __init__(self, base_dir: str = _DEFAULT_BASE_DIR) -> None:
        self.base_dir = str(Path(base_dir).expanduser())

    # ── BaseTool interface ────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "table"

    @property
    def description(self) -> str:
        return (
            "Read, filter, aggregate, and write CSV/TSV tabular data"
            " without pandas. Auto-detects delimiters."
        )

    def definitions(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "table_read",
                "description": (
                    "Read a CSV or TSV file and return all rows as a JSON array of objects."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "filepath": {
                            "type": "string",
                            "description": "CSV/TSV file path",
                        },
                    },
                    "required": ["filepath"],
                },
            },
            {
                "name": "table_columns",
                "description": "Return the column names from the header row of a CSV/TSV file.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "filepath": {
                            "type": "string",
                            "description": "CSV/TSV file path",
                        },
                    },
                    "required": ["filepath"],
                },
            },
            {
                "name": "table_filter",
                "description": (
                    "Filter rows in a CSV/TSV file where a column matches a condition. "
                    "Returns matching rows as a JSON array. "
                    "Operators: eq, ne, gt, lt, gte, lte, contains, startswith."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "filepath": {
                            "type": "string",
                            "description": "CSV/TSV file path",
                        },
                        "column": {
                            "type": "string",
                            "description": "Column name to filter on.",
                        },
                        "operator": {
                            "type": "string",
                            "description": (
                                "Comparison operator: eq, ne, gt, lt, gte, lte, "
                                "contains, startswith."
                            ),
                        },
                        "value": {
                            "type": "string",
                            "description": "Value to compare against.",
                        },
                    },
                    "required": ["filepath", "column", "operator", "value"],
                },
            },
            {
                "name": "table_aggregate",
                "description": (
                    "Compute an aggregate over a column in a CSV/TSV file. "
                    "Operations: count, sum, avg, min, max, unique."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "filepath": {
                            "type": "string",
                            "description": "CSV/TSV file path",
                        },
                        "column": {
                            "type": "string",
                            "description": "Column name to aggregate.",
                        },
                        "operation": {
                            "type": "string",
                            "description": "Aggregation: count, sum, avg, min, max, unique.",
                        },
                    },
                    "required": ["filepath", "column", "operation"],
                },
            },
            {
                "name": "table_write",
                "description": (
                    "Write rows to a CSV file. Rows must be a JSON array of objects "
                    "(string). Returns the number of rows written."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "filepath": {
                            "type": "string",
                            "description": "Destination file path.",
                        },
                        "rows": {
                            "type": "string",
                            "description": "JSON array of row objects to write.",
                        },
                        "delimiter": {
                            "type": "string",
                            "description": "Delimiter character (default: comma).",
                        },
                    },
                    "required": ["filepath", "rows"],
                },
            },
        ]

    def execute(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        if tool_name == "table_read":
            return self._dispatch_read(arguments["filepath"])
        if tool_name == "table_columns":
            return self._dispatch_columns(arguments["filepath"])
        if tool_name == "table_filter":
            return self._dispatch_filter(
                arguments["filepath"],
                arguments["column"],
                arguments["operator"],
                arguments["value"],
            )
        if tool_name == "table_aggregate":
            return self._dispatch_aggregate(
                arguments["filepath"],
                arguments["column"],
                arguments["operation"],
            )
        if tool_name == "table_write":
            return self._dispatch_write(
                arguments["filepath"],
                arguments["rows"],
                arguments.get("delimiter", ","),
            )
        return f"Unknown table tool: {tool_name}"

    # ── Python API ────────────────────────────────────────────────────────────

    def read(self, filepath: str) -> List[Dict[str, str]]:
        """Read CSV/TSV file. Returns list of row dicts. Auto-detects delimiter."""
        path = self._resolve(filepath)
        delimiter = self._detect_delimiter(path)
        with open(path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh, delimiter=delimiter)
            return [dict(row) for row in reader]

    def write(self, filepath: str, rows: List[Dict[str, Any]], delimiter: str = ",") -> int:
        """Write rows to CSV. Returns row count."""
        if not rows:
            return 0
        path = self._resolve(filepath)
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        fieldnames = list(rows[0].keys())
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames, delimiter=delimiter)
            writer.writeheader()
            writer.writerows(rows)
        return len(rows)

    def columns(self, filepath: str) -> List[str]:
        """Return column names from header row."""
        path = self._resolve(filepath)
        delimiter = self._detect_delimiter(path)
        with open(path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh, delimiter=delimiter)
            return list(reader.fieldnames or [])

    def filter_rows(
        self,
        filepath: str,
        column: str,
        operator: str,
        value: str,
    ) -> List[Dict[str, str]]:
        """Filter rows where column op value.

        operator: "eq", "ne", "gt", "lt", "gte", "lte", "contains", "startswith"
        All comparisons are string-based, but try numeric compare when both
        sides are numeric for gt/lt/gte/lte.
        """
        rows = self.read(filepath)
        result = []
        for row in rows:
            cell = row.get(column, "")
            if self._matches(cell, operator, value):
                result.append(row)
        return result

    def aggregate(self, filepath: str, column: str, operation: str) -> str:
        """Aggregate a column. operation: count, sum, avg, min, max, unique.

        Returns string result.
        """
        rows = self.read(filepath)
        values = [row.get(column, "") for row in rows]

        if operation == "count":
            return str(len(values))

        if operation == "unique":
            return str(len(set(values)))

        # numeric operations: skip empty / non-numeric
        numeric = []
        for v in values:
            if v == "":
                continue
            try:
                numeric.append(float(v))
            except ValueError:
                continue

        if operation == "sum":
            return str(sum(numeric)) if numeric else "0"

        if operation == "avg":
            if not numeric:
                return "0"
            return str(statistics.mean(numeric))

        if operation == "min":
            if not numeric:
                # fall back to string min
                non_empty = [v for v in values if v != ""]
                return min(non_empty) if non_empty else ""
            return str(min(numeric))

        if operation == "max":
            if not numeric:
                non_empty = [v for v in values if v != ""]
                return max(non_empty) if non_empty else ""
            return str(max(numeric))

        return f"Unknown operation: {operation}"

    def append_row(self, filepath: str, row: Dict[str, Any]) -> int:
        """Append a single row to existing CSV. Returns new row count."""
        path = self._resolve(filepath)
        delimiter = self._detect_delimiter(path)
        # Read existing to get fieldnames and current count
        existing = self.read(filepath)
        fieldnames = list(existing[0].keys()) if existing else list(row.keys())
        with open(path, "a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames, delimiter=delimiter)
            writer.writerow(row)
        return len(existing) + 1

    # ── path helpers ──────────────────────────────────────────────────────────

    def _resolve(self, filepath: str) -> str:
        """Expand ~ and prepend base_dir for relative paths."""
        if filepath.startswith("~"):
            return str(Path(filepath).expanduser())
        if os.path.isabs(filepath):
            return filepath
        # relative — prepend base_dir
        return os.path.join(self.base_dir, filepath)

    def _detect_delimiter(self, path: str) -> str:
        """Auto-detect delimiter: .tsv → tab, else try comma (check >1 col), fallback tab."""
        if path.endswith(".tsv"):
            return "\t"
        # Try comma first
        try:
            with open(path, newline="", encoding="utf-8") as fh:
                first_line = fh.readline()
            if first_line and "," in first_line:
                # Check that comma produces >1 column
                reader = csv.reader([first_line], delimiter=",")
                row = next(reader, [])
                if len(row) > 1:
                    return ","
        except OSError:
            pass
        return "\t"

    # ── comparison helper ─────────────────────────────────────────────────────

    def _matches(self, cell: str, operator: str, value: str) -> bool:
        """Return True if cell matches the condition."""
        if operator == "contains":
            return value in cell
        if operator == "startswith":
            return cell.startswith(value)
        if operator == "eq":
            return cell == value
        if operator == "ne":
            return cell != value

        # Ordered comparisons — try numeric first
        if operator in ("gt", "lt", "gte", "lte"):
            try:
                cell_n = float(cell)
                value_n = float(value)
                if operator == "gt":
                    return cell_n > value_n
                if operator == "lt":
                    return cell_n < value_n
                if operator == "gte":
                    return cell_n >= value_n
                if operator == "lte":
                    return cell_n <= value_n
            except ValueError:
                # fallback to string compare
                if operator == "gt":
                    return cell > value
                if operator == "lt":
                    return cell < value
                if operator == "gte":
                    return cell >= value
                if operator == "lte":
                    return cell <= value

        return False

    # ── tool-dispatch helpers ─────────────────────────────────────────────────

    def _dispatch_read(self, filepath: str) -> str:
        try:
            rows = self.read(filepath)
        except Exception as exc:
            return f"Error: {exc}"
        return json.dumps(rows)

    def _dispatch_columns(self, filepath: str) -> str:
        try:
            cols = self.columns(filepath)
        except Exception as exc:
            return f"Error: {exc}"
        return json.dumps(cols)

    def _dispatch_filter(
        self, filepath: str, column: str, operator: str, value: str
    ) -> str:
        try:
            rows = self.filter_rows(filepath, column, operator, value)
        except Exception as exc:
            return f"Error: {exc}"
        return json.dumps(rows)

    def _dispatch_aggregate(self, filepath: str, column: str, operation: str) -> str:
        try:
            return self.aggregate(filepath, column, operation)
        except Exception as exc:
            return f"Error: {exc}"

    def _dispatch_write(self, filepath: str, rows_json: str, delimiter: str) -> str:
        try:
            rows = json.loads(rows_json)
        except json.JSONDecodeError as exc:
            return f"Error: invalid JSON — {exc}"
        try:
            count = self.write(filepath, rows, delimiter)
        except Exception as exc:
            return f"Error: {exc}"
        return str(count)
