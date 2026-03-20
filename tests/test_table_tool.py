"""Tests for TableTool."""

import json
import pytest

from agent_friend.tools.table import TableTool


# ── fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture
def tool(tmp_path):
    """Return a TableTool with base_dir set to tmp_path."""
    return TableTool(base_dir=str(tmp_path))


@pytest.fixture
def csv_file(tmp_path):
    """Write a simple CSV and return its path."""
    p = tmp_path / "people.csv"
    p.write_text("name,age,city\nAlice,30,NYC\nBob,25,LA\nCarol,35,NYC\n")
    return p


@pytest.fixture
def tsv_file(tmp_path):
    """Write a simple TSV and return its path."""
    p = tmp_path / "data.tsv"
    p.write_text("name\tage\tcity\nAlice\t30\tNYC\nBob\t25\tLA\n")
    return p


@pytest.fixture
def numeric_csv(tmp_path):
    """CSV with numeric column for aggregation tests."""
    p = tmp_path / "nums.csv"
    p.write_text("label,value\na,10\nb,20\nc,30\nd,\ne,abc\n")
    return p


# ── identity / metadata ────────────────────────────────────────────────────────


def test_name(tool):
    assert tool.name == "table"


def test_description(tool):
    desc = tool.description.lower()
    assert "csv" in desc or "table" in desc or "tabular" in desc


def test_definitions_count(tool):
    defs = tool.definitions()
    assert len(defs) == 5


def test_definitions_names(tool):
    names = {d["name"] for d in tool.definitions()}
    assert names == {
        "table_read",
        "table_columns",
        "table_filter",
        "table_aggregate",
        "table_write",
    }


# ── read ───────────────────────────────────────────────────────────────────────


def test_read_csv(csv_file):
    tool = TableTool()
    rows = tool.read(str(csv_file))
    assert len(rows) == 3
    assert rows[0] == {"name": "Alice", "age": "30", "city": "NYC"}


def test_read_tsv(tsv_file):
    tool = TableTool()
    rows = tool.read(str(tsv_file))
    assert len(rows) == 2
    assert rows[0]["name"] == "Alice"
    assert rows[1]["age"] == "25"


def test_read_empty_file(tmp_path):
    """A file with only a header (no data rows) returns empty list."""
    p = tmp_path / "empty.csv"
    p.write_text("name,age\n")
    tool = TableTool()
    rows = tool.read(str(p))
    assert rows == []


def test_read_single_row(tmp_path):
    p = tmp_path / "single.csv"
    p.write_text("x,y\n1,2\n")
    tool = TableTool()
    rows = tool.read(str(p))
    assert len(rows) == 1
    assert rows[0] == {"x": "1", "y": "2"}


# ── write ──────────────────────────────────────────────────────────────────────


def test_write_and_read_back(tmp_path):
    tool = TableTool()
    out = tmp_path / "out.csv"
    rows = [{"name": "Alice", "score": "95"}, {"name": "Bob", "score": "88"}]
    count = tool.write(str(out), rows)
    assert count == 2
    back = tool.read(str(out))
    assert back == rows


def test_write_returns_row_count(tmp_path):
    tool = TableTool()
    out = tmp_path / "out.csv"
    rows = [{"a": str(i)} for i in range(10)]
    assert tool.write(str(out), rows) == 10


def test_write_empty_rows(tmp_path):
    tool = TableTool()
    out = tmp_path / "empty_out.csv"
    assert tool.write(str(out), []) == 0


# ── columns ────────────────────────────────────────────────────────────────────


def test_columns(csv_file):
    tool = TableTool()
    cols = tool.columns(str(csv_file))
    assert cols == ["name", "age", "city"]


def test_columns_tsv(tsv_file):
    tool = TableTool()
    cols = tool.columns(str(tsv_file))
    assert cols == ["name", "age", "city"]


# ── filter_rows ────────────────────────────────────────────────────────────────


def test_filter_eq(csv_file):
    tool = TableTool()
    rows = tool.filter_rows(str(csv_file), "city", "eq", "NYC")
    assert len(rows) == 2
    assert all(r["city"] == "NYC" for r in rows)


def test_filter_ne(csv_file):
    tool = TableTool()
    rows = tool.filter_rows(str(csv_file), "city", "ne", "NYC")
    assert len(rows) == 1
    assert rows[0]["name"] == "Bob"


def test_filter_contains(csv_file):
    tool = TableTool()
    rows = tool.filter_rows(str(csv_file), "name", "contains", "ol")
    assert len(rows) == 1
    assert rows[0]["name"] == "Carol"


def test_filter_startswith(csv_file):
    tool = TableTool()
    rows = tool.filter_rows(str(csv_file), "name", "startswith", "A")
    assert len(rows) == 1
    assert rows[0]["name"] == "Alice"


def test_filter_gt_numeric(csv_file):
    tool = TableTool()
    rows = tool.filter_rows(str(csv_file), "age", "gt", "29")
    assert len(rows) == 2
    names = {r["name"] for r in rows}
    assert "Alice" in names and "Carol" in names


def test_filter_lt_numeric(csv_file):
    tool = TableTool()
    rows = tool.filter_rows(str(csv_file), "age", "lt", "30")
    assert len(rows) == 1
    assert rows[0]["name"] == "Bob"


def test_filter_gte(csv_file):
    tool = TableTool()
    rows = tool.filter_rows(str(csv_file), "age", "gte", "30")
    assert len(rows) == 2


def test_filter_lte(csv_file):
    tool = TableTool()
    rows = tool.filter_rows(str(csv_file), "age", "lte", "30")
    assert len(rows) == 2


def test_filter_no_matches(csv_file):
    tool = TableTool()
    rows = tool.filter_rows(str(csv_file), "city", "eq", "London")
    assert rows == []


# ── aggregate ──────────────────────────────────────────────────────────────────


def test_aggregate_count(numeric_csv):
    tool = TableTool()
    result = tool.aggregate(str(numeric_csv), "value", "count")
    # count includes empty and non-numeric rows too
    assert result == "5"


def test_aggregate_sum(numeric_csv):
    tool = TableTool()
    result = tool.aggregate(str(numeric_csv), "value", "sum")
    assert float(result) == pytest.approx(60.0)


def test_aggregate_avg(numeric_csv):
    tool = TableTool()
    result = tool.aggregate(str(numeric_csv), "value", "avg")
    assert float(result) == pytest.approx(20.0)


def test_aggregate_min(numeric_csv):
    tool = TableTool()
    result = tool.aggregate(str(numeric_csv), "value", "min")
    assert float(result) == pytest.approx(10.0)


def test_aggregate_max(numeric_csv):
    tool = TableTool()
    result = tool.aggregate(str(numeric_csv), "value", "max")
    assert float(result) == pytest.approx(30.0)


def test_aggregate_unique(csv_file):
    tool = TableTool()
    result = tool.aggregate(str(csv_file), "city", "unique")
    assert result == "2"


def test_aggregate_count_all_rows(csv_file):
    tool = TableTool()
    result = tool.aggregate(str(csv_file), "name", "count")
    assert result == "3"


def test_aggregate_empty_column_values_skipped(tmp_path):
    """sum/avg should skip empty values without error."""
    p = tmp_path / "sparse.csv"
    p.write_text("x\n1\n\n3\n\n")
    tool = TableTool()
    assert float(tool.aggregate(str(p), "x", "sum")) == pytest.approx(4.0)
    assert float(tool.aggregate(str(p), "x", "avg")) == pytest.approx(2.0)


# ── append_row ─────────────────────────────────────────────────────────────────


def test_append_row(csv_file):
    tool = TableTool()
    new_count = tool.append_row(str(csv_file), {"name": "Dave", "age": "40", "city": "SF"})
    assert new_count == 4
    rows = tool.read(str(csv_file))
    assert rows[-1]["name"] == "Dave"


def test_append_row_returns_new_count(tmp_path):
    p = tmp_path / "a.csv"
    p.write_text("k,v\n1,a\n")
    tool = TableTool()
    count = tool.append_row(str(p), {"k": "2", "v": "b"})
    assert count == 2


# ── relative path / base_dir ───────────────────────────────────────────────────


def test_relative_path_uses_base_dir(tmp_path):
    """Relative filepath should be resolved under base_dir."""
    tool = TableTool(base_dir=str(tmp_path))
    rows = [{"col": "val"}]
    tool.write("relative.csv", rows)
    expected = tmp_path / "relative.csv"
    assert expected.exists()
    back = tool.read("relative.csv")
    assert back == rows


# ── tool dispatch: table_read ──────────────────────────────────────────────────


def test_tool_call_table_read(csv_file):
    tool = TableTool()
    result = tool.execute("table_read", {"filepath": str(csv_file)})
    rows = json.loads(result)
    assert isinstance(rows, list)
    assert len(rows) == 3
    assert rows[0]["name"] == "Alice"


def test_tool_call_table_read_returns_string(csv_file):
    tool = TableTool()
    result = tool.execute("table_read", {"filepath": str(csv_file)})
    assert isinstance(result, str)


def test_tool_call_table_columns(csv_file):
    tool = TableTool()
    result = tool.execute("table_columns", {"filepath": str(csv_file)})
    cols = json.loads(result)
    assert cols == ["name", "age", "city"]


# ── tool dispatch: table_filter ────────────────────────────────────────────────


def test_tool_call_table_filter(csv_file):
    tool = TableTool()
    result = tool.execute(
        "table_filter",
        {"filepath": str(csv_file), "column": "city", "operator": "eq", "value": "NYC"},
    )
    rows = json.loads(result)
    assert len(rows) == 2


def test_tool_call_table_filter_no_matches(csv_file):
    tool = TableTool()
    result = tool.execute(
        "table_filter",
        {"filepath": str(csv_file), "column": "city", "operator": "eq", "value": "Tokyo"},
    )
    rows = json.loads(result)
    assert rows == []


# ── tool dispatch: table_aggregate ────────────────────────────────────────────


def test_tool_call_table_aggregate(csv_file):
    tool = TableTool()
    result = tool.execute(
        "table_aggregate",
        {"filepath": str(csv_file), "column": "city", "operation": "unique"},
    )
    assert result == "2"


def test_tool_call_table_aggregate_count(csv_file):
    tool = TableTool()
    result = tool.execute(
        "table_aggregate",
        {"filepath": str(csv_file), "column": "name", "operation": "count"},
    )
    assert result == "3"


# ── tool dispatch: table_write ────────────────────────────────────────────────


def test_tool_call_table_write(tmp_path):
    tool = TableTool()
    out = tmp_path / "written.csv"
    rows = [{"x": "1", "y": "a"}, {"x": "2", "y": "b"}]
    result = tool.execute(
        "table_write",
        {"filepath": str(out), "rows": json.dumps(rows)},
    )
    assert result == "2"
    back = tool.read(str(out))
    assert back == rows


def test_tool_call_table_write_tsv(tmp_path):
    tool = TableTool()
    out = tmp_path / "written.tsv"
    rows = [{"a": "1", "b": "2"}]
    result = tool.execute(
        "table_write",
        {"filepath": str(out), "rows": json.dumps(rows), "delimiter": "\t"},
    )
    assert result == "1"
    content = out.read_text()
    assert "\t" in content


def test_tool_call_table_write_bad_json(tmp_path):
    tool = TableTool()
    out = tmp_path / "bad.csv"
    result = tool.execute(
        "table_write",
        {"filepath": str(out), "rows": "not-json"},
    )
    assert "Error" in result


# ── unknown tool ───────────────────────────────────────────────────────────────


def test_dispatch_unknown_tool(tool):
    result = tool.execute("table_unknown", {})
    assert "Unknown" in result
