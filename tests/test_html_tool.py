"""Tests for HTMLTool."""

import json
import pytest

from agent_friend.tools.html_tool import HTMLTool


@pytest.fixture
def tool():
    return HTMLTool()


SIMPLE_HTML = """
<!DOCTYPE html>
<html>
<head>
  <title>Test Page</title>
  <meta name="description" content="A test page">
  <meta property="og:title" content="OG Title">
</head>
<body>
  <h1>Main Heading</h1>
  <h2>Sub Heading</h2>
  <p>Hello world. This is a <a href="/about">test link</a>.</p>
  <p>Another <a href="https://example.com">external link</a> here.</p>
  <script>var x = 1;</script>
  <style>body { color: red; }</style>
</body>
</html>
"""

TABLE_HTML = """
<table>
  <tr><th>Name</th><th>Age</th></tr>
  <tr><td>Alice</td><td>30</td></tr>
  <tr><td>Bob</td><td>25</td></tr>
</table>
<table>
  <tr><td>Only</td><td>One</td></tr>
</table>
"""


# ── basic properties ───────────────────────────────────────────────────────────


def test_name(tool):
    assert tool.name == "html"


def test_description(tool):
    assert "html" in tool.description.lower()


def test_definitions_count(tool):
    assert len(tool.definitions()) == 6


def test_definition_names(tool):
    names = {d["name"] for d in tool.definitions()}
    assert names == {"html_text", "html_links", "html_headings", "html_meta", "html_tables", "html_select"}


# ── html_text ──────────────────────────────────────────────────────────────────


def test_text_extracts_visible(tool):
    text = tool.html_text(SIMPLE_HTML)
    assert "Hello world" in text
    assert "test link" in text


def test_text_skips_script(tool):
    text = tool.html_text(SIMPLE_HTML)
    assert "var x = 1" not in text


def test_text_skips_style(tool):
    text = tool.html_text(SIMPLE_HTML)
    assert "color: red" not in text


def test_text_skips_tags(tool):
    text = tool.html_text(SIMPLE_HTML)
    assert "<p>" not in text
    assert "<h1>" not in text


def test_text_simple_string(tool):
    text = tool.html_text("<p>Hello</p><p>World</p>")
    assert "Hello" in text
    assert "World" in text


def test_text_empty(tool):
    text = tool.html_text("")
    assert text == ""


def test_text_plain_text_passthrough(tool):
    text = tool.html_text("just plain text")
    assert "just plain text" in text


def test_text_max_chars(tool):
    long_html = "<p>" + "a" * 100_000 + "</p>"
    text = tool.html_text(long_html, max_chars=1000)
    assert len(text) <= 1100  # allows for truncation suffix


def test_text_truncation_message(tool):
    long_html = "<p>" + "x" * 30_000 + "</p>"
    text = tool.html_text(long_html, max_chars=100)
    assert "truncated" in text


def test_text_heading_text_included(tool):
    text = tool.html_text(SIMPLE_HTML)
    assert "Main Heading" in text
    assert "Sub Heading" in text


def test_text_html_entities(tool):
    text = tool.html_text("<p>Tom &amp; Jerry &lt;3</p>")
    assert "&" in text or "Tom" in text


def test_text_default_max_chars_respected(tool):
    huge = "<p>" + "z" * 50_000 + "</p>"
    text = tool.html_text(huge)
    assert len(text) <= 21_000  # 20000 + small suffix


# ── html_links ─────────────────────────────────────────────────────────────────


def test_links_count(tool):
    links = tool.html_links(SIMPLE_HTML)
    assert len(links) == 2


def test_links_href(tool):
    links = tool.html_links(SIMPLE_HTML)
    hrefs = [l["href"] for l in links]
    assert "/about" in hrefs
    assert "https://example.com" in hrefs


def test_links_text(tool):
    links = tool.html_links(SIMPLE_HTML)
    texts = [l["text"] for l in links]
    assert "test link" in texts
    assert "external link" in texts


def test_links_empty_html(tool):
    assert tool.html_links("") == []


def test_links_no_anchors(tool):
    assert tool.html_links("<p>No links here.</p>") == []


def test_links_base_url_absolute_path(tool):
    html = '<a href="/docs">Docs</a>'
    links = tool.html_links(html, base_url="https://example.com/page")
    assert links[0]["href"] == "https://example.com/docs"


def test_links_base_url_relative_path(tool):
    html = '<a href="docs">Docs</a>'
    links = tool.html_links(html, base_url="https://example.com/base")
    assert "docs" in links[0]["href"]


def test_links_base_url_absolute_href_unchanged(tool):
    html = '<a href="https://other.com">Link</a>'
    links = tool.html_links(html, base_url="https://example.com")
    assert links[0]["href"] == "https://other.com"


def test_links_no_href_skipped(tool):
    html = '<a>No href</a><a href="/path">Has href</a>'
    links = tool.html_links(html)
    assert len(links) == 1
    assert links[0]["href"] == "/path"


# ── html_headings ──────────────────────────────────────────────────────────────


def test_headings_count(tool):
    headings = tool.html_headings(SIMPLE_HTML)
    assert len(headings) == 2


def test_headings_levels(tool):
    headings = tool.html_headings(SIMPLE_HTML)
    assert headings[0]["level"] == 1
    assert headings[1]["level"] == 2


def test_headings_text(tool):
    headings = tool.html_headings(SIMPLE_HTML)
    assert headings[0]["text"] == "Main Heading"
    assert headings[1]["text"] == "Sub Heading"


def test_headings_all_levels(tool):
    html = "".join(f"<h{i}>H{i}</h{i}>" for i in range(1, 7))
    headings = tool.html_headings(html)
    assert len(headings) == 6
    for i, h in enumerate(headings, 1):
        assert h["level"] == i


def test_headings_empty(tool):
    assert tool.html_headings("<p>No headings</p>") == []


def test_headings_empty_html(tool):
    assert tool.html_headings("") == []


# ── html_meta ─────────────────────────────────────────────────────────────────


def test_meta_title(tool):
    result = tool.html_meta(SIMPLE_HTML)
    assert result["title"] == "Test Page"


def test_meta_tags_present(tool):
    result = tool.html_meta(SIMPLE_HTML)
    assert len(result["meta"]) >= 2


def test_meta_description(tool):
    result = tool.html_meta(SIMPLE_HTML)
    names = [m.get("name") for m in result["meta"]]
    assert "description" in names


def test_meta_og_title(tool):
    result = tool.html_meta(SIMPLE_HTML)
    props = [m.get("property") for m in result["meta"]]
    assert "og:title" in props


def test_meta_no_title(tool):
    result = tool.html_meta("<p>No title tag</p>")
    assert result["title"] == ""


def test_meta_empty_html(tool):
    result = tool.html_meta("")
    assert result["title"] == ""
    assert result["meta"] == []


# ── html_tables ───────────────────────────────────────────────────────────────


def test_tables_count(tool):
    tables = tool.html_tables(TABLE_HTML)
    assert len(tables) == 2


def test_tables_first_table_rows(tool):
    tables = tool.html_tables(TABLE_HTML)
    assert len(tables[0]) == 3  # header + 2 data rows


def test_tables_header_row(tool):
    tables = tool.html_tables(TABLE_HTML)
    assert tables[0][0] == ["Name", "Age"]


def test_tables_data_rows(tool):
    tables = tool.html_tables(TABLE_HTML)
    assert tables[0][1] == ["Alice", "30"]
    assert tables[0][2] == ["Bob", "25"]


def test_tables_second_table(tool):
    tables = tool.html_tables(TABLE_HTML)
    assert tables[1][0] == ["Only", "One"]


def test_tables_no_tables(tool):
    assert tool.html_tables("<p>No tables here</p>") == []


def test_tables_empty_html(tool):
    assert tool.html_tables("") == []


# ── html_select ───────────────────────────────────────────────────────────────


def test_select_by_tag(tool):
    html = "<p>One</p><p>Two</p><p>Three</p>"
    results = tool.html_select(html, "p")
    assert results == ["One", "Two", "Three"]


def test_select_code_blocks(tool):
    html = "<pre><code>print('hello')</code></pre><code>x = 1</code>"
    results = tool.html_select(html, "code")
    assert len(results) == 2
    assert any("print" in r for r in results)


def test_select_with_attrs(tool):
    html = '<span class="price">$10</span><span class="label">Name</span><span class="price">$20</span>'
    results = tool.html_select(html, "span", {"class": "price"})
    assert results == ["$10", "$20"]


def test_select_no_match(tool):
    results = tool.html_select("<p>Hello</p>", "div")
    assert results == []


def test_select_attrs_no_match(tool):
    html = '<span class="other">text</span>'
    results = tool.html_select(html, "span", {"class": "price"})
    assert results == []


def test_select_empty_html(tool):
    assert tool.html_select("", "p") == []


# ── execute dispatch ───────────────────────────────────────────────────────────


def test_execute_html_text(tool):
    result = tool.execute("html_text", {"html": "<p>Hello</p>"})
    assert "Hello" in result


def test_execute_html_links(tool):
    result = json.loads(tool.execute("html_links", {"html": '<a href="/x">X</a>'}))
    assert result[0]["href"] == "/x"


def test_execute_html_headings(tool):
    result = json.loads(tool.execute("html_headings", {"html": "<h1>Title</h1>"}))
    assert result[0]["text"] == "Title"


def test_execute_html_meta(tool):
    result = json.loads(tool.execute("html_meta", {"html": "<title>T</title>"}))
    assert result["title"] == "T"


def test_execute_html_tables(tool):
    html = "<table><tr><td>A</td><td>B</td></tr></table>"
    result = json.loads(tool.execute("html_tables", {"html": html}))
    assert result[0][0] == ["A", "B"]


def test_execute_html_select(tool):
    result = json.loads(tool.execute("html_select", {"html": "<p>Hi</p>", "tag": "p"}))
    assert result == ["Hi"]


def test_execute_unknown(tool):
    result = json.loads(tool.execute("unknown_tool", {}))
    assert "error" in result
