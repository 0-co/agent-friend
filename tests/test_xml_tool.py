"""Tests for XMLTool."""

import json
import pytest

from agent_friend.tools.xml_tool import XMLTool


SIMPLE_XML = """<?xml version="1.0"?>
<catalog>
  <book id="1" lang="en">
    <title>Agent Patterns</title>
    <author>Alice</author>
    <price>29.99</price>
  </book>
  <book id="2" lang="fr">
    <title>Agents en Python</title>
    <author>Bob</author>
    <price>24.99</price>
  </book>
</catalog>"""

FLAT_XML = "<root><item>A</item><item>B</item><item>C</item></root>"

NS_XML = """<?xml version="1.0"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>Test Feed</title>
  <entry><title>Entry One</title></entry>
  <entry><title>Entry Two</title></entry>
</feed>"""

MALFORMED_XML = "<root><unclosed>"


@pytest.fixture
def tool():
    return XMLTool()


# ── basic properties ───────────────────────────────────────────────────────────


def test_name(tool):
    assert tool.name == "xml"


def test_description(tool):
    assert "xml" in tool.description.lower()


def test_definitions_count(tool):
    assert len(tool.definitions()) == 7


def test_definition_names(tool):
    names = {d["name"] for d in tool.definitions()}
    assert names == {
        "xml_extract", "xml_attrs", "xml_find", "xml_findall",
        "xml_to_dict", "xml_validate", "xml_tags",
    }


# ── xml_extract ────────────────────────────────────────────────────────────────


def test_extract_titles(tool):
    result = json.loads(tool.xml_extract(SIMPLE_XML, "title"))
    assert result == ["Agent Patterns", "Agents en Python"]


def test_extract_authors(tool):
    result = json.loads(tool.xml_extract(SIMPLE_XML, "author"))
    assert result == ["Alice", "Bob"]


def test_extract_flat(tool):
    result = json.loads(tool.xml_extract(FLAT_XML, "item"))
    assert result == ["A", "B", "C"]


def test_extract_missing_tag(tool):
    result = json.loads(tool.xml_extract(SIMPLE_XML, "publisher"))
    assert result == []


def test_extract_malformed(tool):
    result = json.loads(tool.xml_extract(MALFORMED_XML, "root"))
    assert "error" in result


def test_extract_max_results(tool):
    xml = "<r>" + "<x>v</x>" * 200 + "</r>"
    t = XMLTool(max_results=10)
    result = json.loads(t.xml_extract(xml, "x"))
    assert len(result) == 10


def test_extract_root_tag(tool):
    result = json.loads(tool.xml_extract("<root>text</root>", "root"))
    assert result == ["text"]


# ── xml_attrs ─────────────────────────────────────────────────────────────────


def test_attrs_book_ids(tool):
    result = json.loads(tool.xml_attrs(SIMPLE_XML, "book"))
    assert len(result) == 2
    assert result[0]["id"] == "1"
    assert result[1]["id"] == "2"


def test_attrs_lang(tool):
    result = json.loads(tool.xml_attrs(SIMPLE_XML, "book"))
    assert result[0]["lang"] == "en"
    assert result[1]["lang"] == "fr"


def test_attrs_no_attrs(tool):
    result = json.loads(tool.xml_attrs(SIMPLE_XML, "title"))
    assert result == [{}, {}]


def test_attrs_malformed(tool):
    result = json.loads(tool.xml_attrs(MALFORMED_XML, "root"))
    assert "error" in result


def test_attrs_missing_tag(tool):
    result = json.loads(tool.xml_attrs(SIMPLE_XML, "missing"))
    assert result == []


# ── xml_find ──────────────────────────────────────────────────────────────────


def test_find_first_book(tool):
    result = json.loads(tool.xml_find(SIMPLE_XML, ".//book"))
    assert result["found"] is True
    assert result["tag"] == "book"
    assert result["attrs"]["id"] == "1"


def test_find_by_attr(tool):
    result = json.loads(tool.xml_find(SIMPLE_XML, ".//book[@id='2']"))
    assert result["found"] is True
    assert result["attrs"]["id"] == "2"


def test_find_title(tool):
    result = json.loads(tool.xml_find(SIMPLE_XML, ".//title"))
    assert result["text"] == "Agent Patterns"


def test_find_not_found(tool):
    result = json.loads(tool.xml_find(SIMPLE_XML, ".//missing"))
    assert result["found"] is False


def test_find_children(tool):
    result = json.loads(tool.xml_find(SIMPLE_XML, ".//book"))
    assert "title" in result["children"]
    assert "author" in result["children"]


def test_find_malformed_xml(tool):
    result = json.loads(tool.xml_find(MALFORMED_XML, ".//x"))
    assert "error" in result


# ── xml_findall ───────────────────────────────────────────────────────────────


def test_findall_books(tool):
    result = json.loads(tool.xml_findall(SIMPLE_XML, ".//book"))
    assert len(result) == 2


def test_findall_returns_attrs(tool):
    result = json.loads(tool.xml_findall(SIMPLE_XML, ".//book"))
    assert result[0]["attrs"]["id"] == "1"
    assert result[1]["attrs"]["id"] == "2"


def test_findall_empty(tool):
    result = json.loads(tool.xml_findall(SIMPLE_XML, ".//missing"))
    assert result == []


def test_findall_malformed(tool):
    result = json.loads(tool.xml_findall(MALFORMED_XML, ".//x"))
    assert "error" in result


def test_findall_max_results(tool):
    xml = "<r>" + "<x/>" * 200 + "</r>"
    t = XMLTool(max_results=5)
    result = json.loads(t.xml_findall(xml, ".//x"))
    assert len(result) == 5


# ── xml_to_dict ───────────────────────────────────────────────────────────────


def test_to_dict_structure(tool):
    result = json.loads(tool.xml_to_dict(FLAT_XML))
    assert "root" in result


def test_to_dict_simple_text(tool):
    result = json.loads(tool.xml_to_dict("<a><b>hello</b></a>"))
    assert result == {"a": {"b": "hello"}}


def test_to_dict_attrs(tool):
    result = json.loads(tool.xml_to_dict('<a x="1">text</a>'))
    a = result["a"]
    assert a["@x"] == "1"


def test_to_dict_repeated_children_become_list(tool):
    result = json.loads(tool.xml_to_dict(FLAT_XML))
    assert isinstance(result["root"]["item"], list)
    assert result["root"]["item"] == ["A", "B", "C"]


def test_to_dict_malformed(tool):
    result = json.loads(tool.xml_to_dict(MALFORMED_XML))
    assert "error" in result


def test_to_dict_nested(tool):
    result = json.loads(tool.xml_to_dict("<r><a><b>x</b></a></r>"))
    assert result["r"]["a"]["b"] == "x"


# ── xml_validate ──────────────────────────────────────────────────────────────


def test_validate_valid(tool):
    result = json.loads(tool.xml_validate(SIMPLE_XML))
    assert result["valid"] is True


def test_validate_flat_valid(tool):
    result = json.loads(tool.xml_validate(FLAT_XML))
    assert result["valid"] is True


def test_validate_malformed(tool):
    result = json.loads(tool.xml_validate(MALFORMED_XML))
    assert result["valid"] is False
    assert "error" in result


def test_validate_empty_root(tool):
    result = json.loads(tool.xml_validate("<empty/>"))
    assert result["valid"] is True


def test_validate_gibberish(tool):
    result = json.loads(tool.xml_validate("not xml at all"))
    assert result["valid"] is False


# ── xml_tags ──────────────────────────────────────────────────────────────────


def test_tags_catalog(tool):
    result = json.loads(tool.xml_tags(SIMPLE_XML))
    assert "catalog" in result
    assert "book" in result
    assert result["book"] == 2


def test_tags_flat(tool):
    result = json.loads(tool.xml_tags(FLAT_XML))
    assert result["root"] == 1
    assert result["item"] == 3


def test_tags_malformed(tool):
    result = json.loads(tool.xml_tags(MALFORMED_XML))
    assert "error" in result


# ── namespace handling ────────────────────────────────────────────────────────


def test_ns_extract_strips_namespace(tool):
    result = json.loads(tool.xml_extract(NS_XML, "title"))
    assert "Test Feed" in result


def test_ns_tags_strips_namespace(tool):
    result = json.loads(tool.xml_tags(NS_XML))
    assert "feed" in result or "title" in result


# ── execute dispatch ───────────────────────────────────────────────────────────


def test_execute_xml_extract(tool):
    result = json.loads(tool.execute("xml_extract", {"xml_string": FLAT_XML, "tag": "item"}))
    assert result == ["A", "B", "C"]


def test_execute_xml_attrs(tool):
    result = json.loads(tool.execute("xml_attrs", {"xml_string": SIMPLE_XML, "tag": "book"}))
    assert len(result) == 2


def test_execute_xml_find(tool):
    result = json.loads(tool.execute("xml_find", {"xml_string": FLAT_XML, "xpath": ".//item"}))
    assert result["found"] is True


def test_execute_xml_findall(tool):
    result = json.loads(tool.execute("xml_findall", {"xml_string": FLAT_XML, "xpath": ".//item"}))
    assert len(result) == 3


def test_execute_xml_to_dict(tool):
    result = json.loads(tool.execute("xml_to_dict", {"xml_string": FLAT_XML}))
    assert "root" in result


def test_execute_xml_validate(tool):
    result = json.loads(tool.execute("xml_validate", {"xml_string": FLAT_XML}))
    assert result["valid"] is True


def test_execute_xml_tags(tool):
    result = json.loads(tool.execute("xml_tags", {"xml_string": FLAT_XML}))
    assert "item" in result


def test_execute_unknown(tool):
    result = json.loads(tool.execute("no_such", {}))
    assert "error" in result
