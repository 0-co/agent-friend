"""Tests for DatabaseTool."""

import sqlite3
import pytest

from agent_friend.tools.database import DatabaseTool


# ── fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def db(tmp_path):
    """Return a DatabaseTool backed by a temp-dir database."""
    return DatabaseTool(db_path=str(tmp_path / "test.db"))


@pytest.fixture
def db_with_tasks(db):
    """DatabaseTool with a tasks table and two rows."""
    db.create_table(
        "tasks",
        "id INTEGER PRIMARY KEY, title TEXT NOT NULL, done INTEGER DEFAULT 0",
    )
    db.insert("tasks", {"title": "Buy groceries", "done": 0})
    db.insert("tasks", {"title": "Write tests", "done": 1})
    return db


# ── init / connectivity ───────────────────────────────────────────────────────


def test_init_creates_db_file(tmp_path):
    db_path = str(tmp_path / "sub" / "agent.db")
    db = DatabaseTool(db_path=db_path)
    import os
    assert os.path.isdir(str(tmp_path / "sub"))


def test_name(db):
    assert db.name == "database"


def test_description(db):
    assert "SQLite" in db.description or "database" in db.description.lower()


def test_definitions_returns_four_tools(db):
    defs = db.definitions()
    assert len(defs) == 4
    names = {d["name"] for d in defs}
    assert names == {"db_query", "db_execute", "db_tables", "db_schema"}


# ── create_table ──────────────────────────────────────────────────────────────


def test_create_table(db):
    db.create_table("items", "id INTEGER PRIMARY KEY, name TEXT")
    assert "items" in db.list_tables()


def test_create_table_idempotent(db):
    db.create_table("items", "id INTEGER PRIMARY KEY, name TEXT")
    db.create_table("items", "id INTEGER PRIMARY KEY, name TEXT")  # no error
    assert db.list_tables().count("items") == 1


def test_create_multiple_tables(db):
    db.create_table("a", "id INTEGER PRIMARY KEY")
    db.create_table("b", "id INTEGER PRIMARY KEY")
    tables = db.list_tables()
    assert "a" in tables and "b" in tables


# ── insert ────────────────────────────────────────────────────────────────────


def test_insert_returns_rowid(db_with_tasks):
    rowid = db_with_tasks.insert("tasks", {"title": "Third", "done": 0})
    assert rowid == 3


def test_insert_row_appears_in_query(db_with_tasks):
    rows = db_with_tasks.query("SELECT title FROM tasks WHERE done = 0")
    titles = [r["title"] for r in rows]
    assert "Buy groceries" in titles


def test_insert_multiple_rows(db):
    db.create_table("nums", "n INTEGER")
    for i in range(5):
        db.insert("nums", {"n": i})
    rows = db.query("SELECT COUNT(*) AS cnt FROM nums")
    assert rows[0]["cnt"] == 5


# ── query ─────────────────────────────────────────────────────────────────────


def test_query_returns_list_of_dicts(db_with_tasks):
    rows = db_with_tasks.query("SELECT * FROM tasks")
    assert isinstance(rows, list)
    assert isinstance(rows[0], dict)


def test_query_with_params(db_with_tasks):
    rows = db_with_tasks.query("SELECT title FROM tasks WHERE done = ?", [1])
    assert rows[0]["title"] == "Write tests"


def test_query_empty_result(db_with_tasks):
    rows = db_with_tasks.query("SELECT * FROM tasks WHERE done = 99")
    assert rows == []


def test_query_column_names_present(db_with_tasks):
    rows = db_with_tasks.query("SELECT id, title, done FROM tasks")
    assert set(rows[0].keys()) == {"id", "title", "done"}


# ── run ───────────────────────────────────────────────────────────────────────


def test_run_update(db_with_tasks):
    rowcount = db_with_tasks.run("UPDATE tasks SET done = 1 WHERE done = 0")
    assert rowcount == 1
    rows = db_with_tasks.query("SELECT * FROM tasks WHERE done = 0")
    assert rows == []


def test_run_delete(db_with_tasks):
    rowcount = db_with_tasks.run("DELETE FROM tasks WHERE id = ?", [1])
    assert rowcount == 1
    rows = db_with_tasks.query("SELECT * FROM tasks")
    assert len(rows) == 1


def test_run_returns_rowcount(db_with_tasks):
    rowcount = db_with_tasks.run("DELETE FROM tasks")
    assert rowcount == 2


def test_run_create_drop(db):
    db.run("CREATE TABLE tmp (x INTEGER)")
    assert "tmp" in db.list_tables()
    db.run("DROP TABLE tmp")
    assert "tmp" not in db.list_tables()


# ── list_tables ───────────────────────────────────────────────────────────────


def test_list_tables_empty(db):
    assert db.list_tables() == []


def test_list_tables_sorted(db):
    db.create_table("zebra", "id INTEGER")
    db.create_table("apple", "id INTEGER")
    tables = db.list_tables()
    assert tables == sorted(tables)


def test_list_tables_excludes_sqlite_internals(db):
    db.create_table("items", "id INTEGER")
    for name in db.list_tables():
        assert not name.startswith("sqlite_")


# ── get_schema ────────────────────────────────────────────────────────────────


def test_get_schema_contains_table_name(db_with_tasks):
    schema = db_with_tasks.get_schema("tasks")
    assert "tasks" in schema


def test_get_schema_contains_columns(db_with_tasks):
    schema = db_with_tasks.get_schema("tasks")
    assert "title" in schema
    assert "done" in schema


def test_get_schema_missing_table(db):
    result = db.get_schema("nonexistent")
    assert "not found" in result.lower()


# ── tool dispatch: db_query ───────────────────────────────────────────────────


def test_dispatch_db_query_returns_string(db_with_tasks):
    result = db_with_tasks.execute("db_query", {"sql": "SELECT * FROM tasks"})
    assert isinstance(result, str)
    assert "tasks" not in result or "title" in result  # has header


def test_dispatch_db_query_header_row(db_with_tasks):
    result = db_with_tasks.execute("db_query", {"sql": "SELECT id, title FROM tasks"})
    assert result.startswith("id, title")


def test_dispatch_db_query_no_rows(db_with_tasks):
    result = db_with_tasks.execute(
        "db_query", {"sql": "SELECT * FROM tasks WHERE done = 99"}
    )
    assert "No rows" in result


def test_dispatch_db_query_bad_sql(db):
    result = db.execute("db_query", {"sql": "SELECT * FROM nonexistent_xyz"})
    assert "Error" in result


def test_dispatch_db_query_with_params(db_with_tasks):
    result = db_with_tasks.execute(
        "db_query", {"sql": "SELECT title FROM tasks WHERE done = ?", "params": [1]}
    )
    assert "Write tests" in result


# ── tool dispatch: db_execute ─────────────────────────────────────────────────


def test_dispatch_db_execute_create(db):
    result = db.execute(
        "db_execute", {"sql": "CREATE TABLE t (n INTEGER)"}
    )
    assert "OK" in result
    assert "t" in db.list_tables()


def test_dispatch_db_execute_insert(db_with_tasks):
    result = db_with_tasks.execute(
        "db_execute",
        {"sql": "INSERT INTO tasks (title, done) VALUES (?, ?)", "params": ["New", 0]},
    )
    assert "OK" in result


def test_dispatch_db_execute_update(db_with_tasks):
    result = db_with_tasks.execute(
        "db_execute", {"sql": "UPDATE tasks SET done = 1 WHERE done = 0"}
    )
    assert "1 row" in result


def test_dispatch_db_execute_bad_sql(db):
    result = db.execute("db_execute", {"sql": "INVALID SQL HERE"})
    assert "Error" in result


# ── tool dispatch: db_tables ──────────────────────────────────────────────────


def test_dispatch_db_tables_empty(db):
    result = db.execute("db_tables", {})
    assert "No tables" in result


def test_dispatch_db_tables_lists(db_with_tasks):
    result = db_with_tasks.execute("db_tables", {})
    assert "tasks" in result


# ── tool dispatch: db_schema ──────────────────────────────────────────────────


def test_dispatch_db_schema_found(db_with_tasks):
    result = db_with_tasks.execute("db_schema", {"table": "tasks"})
    assert "CREATE TABLE" in result


def test_dispatch_db_schema_missing(db):
    result = db.execute("db_schema", {"table": "ghost"})
    assert "not found" in result.lower()


# ── unknown tool ──────────────────────────────────────────────────────────────


def test_dispatch_unknown_tool(db):
    result = db.execute("db_unknown", {})
    assert "Unknown" in result
