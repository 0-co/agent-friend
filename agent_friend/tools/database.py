"""database.py — DatabaseTool for agent-friend (SQLite, no required dependencies).

Agents can create tables, insert rows, run SQL queries, and inspect the
database schema — all backed by a local SQLite file.

Usage::

    tool = DatabaseTool()
    tool.create_table("tasks", "id INTEGER PRIMARY KEY, title TEXT NOT NULL, done INTEGER DEFAULT 0")
    tool.insert("tasks", {"title": "Buy groceries", "done": 0})
    rows = tool.query("SELECT * FROM tasks WHERE done = 0")
    tool.run("UPDATE tasks SET done = 1 WHERE id = ?", [1])
    tool.list_tables()
    tool.get_schema("tasks")
"""

import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import BaseTool


_DEFAULT_DB_PATH = "~/.agent_friend/agent.db"


class DatabaseTool(BaseTool):
    """Create, query, and manage SQLite databases.

    Agents can create tables, insert rows, run arbitrary SQL, and inspect
    the schema — backed by a local SQLite file with no external dependencies.

    Parameters
    ----------
    db_path:
        Path to the SQLite database file.
        Defaults to ``~/.agent_friend/agent.db``.
    """

    def __init__(self, db_path: str = _DEFAULT_DB_PATH) -> None:
        self.db_path = str(Path(db_path).expanduser())
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    @property
    def name(self) -> str:
        return "database"

    @property
    def description(self) -> str:
        return (
            "Create tables, insert/update/delete rows, and run SQL queries"
            " on a local SQLite database."
        )

    def definitions(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "db_query",
                "description": (
                    "Run a SQL SELECT query and return results as a table of rows."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "sql": {
                            "type": "string",
                            "description": "A SQL SELECT statement.",
                        },
                        "params": {
                            "type": "array",
                            "items": {},
                            "description": (
                                "Optional positional parameters for ? placeholders."
                            ),
                        },
                    },
                    "required": ["sql"],
                },
            },
            {
                "name": "db_execute",
                "description": (
                    "Run a SQL statement that modifies the database"
                    " (INSERT, UPDATE, DELETE, CREATE TABLE, DROP TABLE, etc.)."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "sql": {
                            "type": "string",
                            "description": "A SQL statement to execute.",
                        },
                        "params": {
                            "type": "array",
                            "items": {},
                            "description": (
                                "Optional positional parameters for ? placeholders."
                            ),
                        },
                    },
                    "required": ["sql"],
                },
            },
            {
                "name": "db_tables",
                "description": "List all tables in the database.",
                "input_schema": {
                    "type": "object",
                    "properties": {},
                },
            },
            {
                "name": "db_schema",
                "description": "Get the CREATE TABLE statement for a specific table.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "table": {
                            "type": "string",
                            "description": "The name of the table.",
                        },
                    },
                    "required": ["table"],
                },
            },
        ]

    def execute(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        if tool_name == "db_query":
            return self._dispatch_query(arguments["sql"], arguments.get("params", []))
        if tool_name == "db_execute":
            return self._dispatch_execute(
                arguments["sql"], arguments.get("params", [])
            )
        if tool_name == "db_tables":
            return self._dispatch_tables()
        if tool_name == "db_schema":
            return self._dispatch_schema(arguments["table"])
        return f"Unknown database tool: {tool_name}"

    # ── Python API ────────────────────────────────────────────────────────────

    def query(self, sql: str, params: Optional[List] = None) -> List[Dict[str, Any]]:
        """Run a SELECT statement and return rows as a list of dicts."""
        params = params or []
        with self._connect() as conn:
            cursor = conn.execute(sql, params)
            return [dict(row) for row in cursor.fetchall()]

    def run(self, sql: str, params: Optional[List] = None) -> int:
        """Run a write statement (INSERT/UPDATE/DELETE/CREATE/DROP).

        Returns the number of rows affected (``cursor.rowcount``).
        """
        params = params or []
        with self._connect() as conn:
            cursor = conn.execute(sql, params)
            conn.commit()
            return cursor.rowcount

    def insert(self, table: str, row: Dict[str, Any]) -> int:
        """Insert *row* into *table* and return the new rowid."""
        cols = ", ".join(row.keys())
        placeholders = ", ".join("?" * len(row))
        sql = f"INSERT INTO {table} ({cols}) VALUES ({placeholders})"
        with self._connect() as conn:
            cursor = conn.execute(sql, list(row.values()))
            conn.commit()
            return cursor.lastrowid or 0

    def create_table(self, table: str, schema: str) -> None:
        """Create *table* if it does not already exist.

        *schema* is the column definitions string, e.g.:
        ``'id INTEGER PRIMARY KEY, name TEXT NOT NULL'``
        """
        self.run(f"CREATE TABLE IF NOT EXISTS {table} ({schema})")

    def list_tables(self) -> List[str]:
        """Return the names of all user tables (sorted)."""
        rows = self.query(
            "SELECT name FROM sqlite_master"
            " WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            " ORDER BY name"
        )
        return [r["name"] for r in rows]

    def get_schema(self, table: str) -> str:
        """Return the CREATE TABLE SQL for *table*, or an error message."""
        rows = self.query(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name=?",
            [table],
        )
        if not rows:
            return f"Table '{table}' not found."
        return rows[0]["sql"] or ""

    # ── tool-dispatch helpers ─────────────────────────────────────────────────

    def _dispatch_query(self, sql: str, params: List) -> str:
        try:
            rows = self.query(sql, params)
        except sqlite3.Error as exc:
            return f"Error: {exc}"
        if not rows:
            return "No rows returned."
        header = ", ".join(rows[0].keys())
        body = "\n".join(", ".join(str(v) for v in row.values()) for row in rows)
        return f"{header}\n{body}"

    def _dispatch_execute(self, sql: str, params: List) -> str:
        try:
            rowcount = self.run(sql, params)
        except sqlite3.Error as exc:
            return f"Error: {exc}"
        return f"OK ({rowcount} row(s) affected)"

    def _dispatch_tables(self) -> str:
        tables = self.list_tables()
        if not tables:
            return "No tables found."
        return "\n".join(tables)

    def _dispatch_schema(self, table: str) -> str:
        return self.get_schema(table)
