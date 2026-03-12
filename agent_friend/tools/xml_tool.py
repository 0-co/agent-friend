"""xml_tool.py — XMLTool for agent-friend (stdlib only).

Many APIs still return XML: SOAP services, Atom feeds, configuration
files, and legacy integrations.  XMLTool gives agents a clean interface
to parse, query, and extract data from XML without external dependencies.

Uses ``xml.etree.ElementTree`` from the standard library.

Usage::

    tool = XMLTool()

    xml = "<root><item id='1'>Apple</item><item id='2'>Banana</item></root>"

    tool.xml_extract(xml, "item")
    # ["Apple", "Banana"]

    tool.xml_attrs(xml, "item")
    # [{"id": "1"}, {"id": "2"}]

    tool.xml_find(xml, ".//item[@id='2']")
    # {"tag": "item", "text": "Banana", "attrs": {"id": "2"}}

    tool.xml_to_dict(xml)
    # {"root": {"item": [{"@id": "1", "#text": "Apple"}, {"@id": "2", "#text": "Banana"}]}}
"""

import json
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Union

from .base import BaseTool


def _elem_to_dict(elem: ET.Element) -> Any:
    """Recursively convert an ElementTree element to a dict/str."""
    children = list(elem)
    result: Dict[str, Any] = {}

    # Attributes with @ prefix
    for k, v in elem.attrib.items():
        result[f"@{k}"] = v

    if children:
        child_dicts: Dict[str, Any] = {}
        for child in children:
            tag = child.tag
            # Strip namespace
            if "}" in tag:
                tag = tag.split("}", 1)[1]
            child_val = _elem_to_dict(child)
            if tag in child_dicts:
                existing = child_dicts[tag]
                if isinstance(existing, list):
                    existing.append(child_val)
                else:
                    child_dicts[tag] = [existing, child_val]
            else:
                child_dicts[tag] = child_val
        result.update(child_dicts)
    else:
        text = (elem.text or "").strip()
        if text:
            if result:
                result["#text"] = text
            else:
                return text

    return result or (elem.text or "").strip()


class XMLTool(BaseTool):
    """Parse XML and extract tags, attributes, and text using XPath-like queries.

    Supports any well-formed XML including SOAP responses, Atom/RSS feeds,
    configuration files, and API payloads.  All stdlib — no lxml required.

    Parameters
    ----------
    max_results:
        Maximum number of results returned by search operations (default 100).
    """

    def __init__(self, max_results: int = 100) -> None:
        self.max_results = max_results

    # ── helpers ───────────────────────────────────────────────────────────

    def _parse(self, xml_string: str) -> ET.Element:
        """Parse *xml_string* and return the root element."""
        return ET.fromstring(xml_string.strip())

    def _strip_ns(self, tag: str) -> str:
        """Strip XML namespace from a tag name like {http://…}name → name."""
        if "}" in tag:
            return tag.split("}", 1)[1]
        return tag

    # ── public API ────────────────────────────────────────────────────────

    def xml_extract(self, xml_string: str, tag: str) -> str:
        """Return text content of all *tag* elements as a JSON list of strings."""
        try:
            root = self._parse(xml_string)
        except ET.ParseError as exc:
            return json.dumps({"error": f"Parse error: {exc}"})

        results = []
        for elem in root.iter():
            if self._strip_ns(elem.tag) == tag:
                text = (elem.text or "").strip()
                results.append(text)
                if len(results) >= self.max_results:
                    break
        return json.dumps(results)

    def xml_attrs(self, xml_string: str, tag: str) -> str:
        """Return attributes of all *tag* elements as a JSON list of dicts."""
        try:
            root = self._parse(xml_string)
        except ET.ParseError as exc:
            return json.dumps({"error": f"Parse error: {exc}"})

        results = []
        for elem in root.iter():
            if self._strip_ns(elem.tag) == tag:
                results.append(dict(elem.attrib))
                if len(results) >= self.max_results:
                    break
        return json.dumps(results)

    def xml_find(self, xml_string: str, xpath: str) -> str:
        """Find the first element matching an XPath expression.

        Returns JSON with: tag, text, attrs, children (list of child tag names).
        """
        try:
            root = self._parse(xml_string)
        except ET.ParseError as exc:
            return json.dumps({"error": f"Parse error: {exc}"})

        try:
            elem = root.find(xpath)
        except (ET.ParseError, SyntaxError) as exc:
            return json.dumps({"error": f"XPath error: {exc}"})

        if elem is None:
            return json.dumps({"found": False})

        return json.dumps({
            "found": True,
            "tag": self._strip_ns(elem.tag),
            "text": (elem.text or "").strip(),
            "attrs": dict(elem.attrib),
            "children": [self._strip_ns(c.tag) for c in elem],
        })

    def xml_findall(self, xml_string: str, xpath: str) -> str:
        """Return all elements matching an XPath expression.

        Each result has: tag, text, attrs.
        """
        try:
            root = self._parse(xml_string)
        except ET.ParseError as exc:
            return json.dumps({"error": f"Parse error: {exc}"})

        try:
            elems = root.findall(xpath)
        except (ET.ParseError, SyntaxError) as exc:
            return json.dumps({"error": f"XPath error: {exc}"})

        results = []
        for elem in elems[: self.max_results]:
            results.append({
                "tag": self._strip_ns(elem.tag),
                "text": (elem.text or "").strip(),
                "attrs": dict(elem.attrib),
            })
        return json.dumps(results)

    def xml_to_dict(self, xml_string: str) -> str:
        """Convert XML to a nested dict/JSON representation.

        Attributes are prefixed with ``@``.  Text content uses ``#text`` when
        the element also has attributes or children.  Repeated sibling tags
        become lists.
        """
        try:
            root = self._parse(xml_string)
        except ET.ParseError as exc:
            return json.dumps({"error": f"Parse error: {exc}"})

        tag = self._strip_ns(root.tag)
        return json.dumps({tag: _elem_to_dict(root)})

    def xml_validate(self, xml_string: str) -> str:
        """Return ``{"valid": true}`` if the XML is well-formed, or an error."""
        try:
            self._parse(xml_string)
            return json.dumps({"valid": True})
        except ET.ParseError as exc:
            return json.dumps({"valid": False, "error": str(exc)})

    def xml_tags(self, xml_string: str) -> str:
        """Return a deduplicated list of all tag names in the document."""
        try:
            root = self._parse(xml_string)
        except ET.ParseError as exc:
            return json.dumps({"error": f"Parse error: {exc}"})

        seen: Dict[str, int] = {}
        for elem in root.iter():
            tag = self._strip_ns(elem.tag)
            seen[tag] = seen.get(tag, 0) + 1
        return json.dumps(seen)

    # ── BaseTool interface ────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "xml"

    @property
    def description(self) -> str:
        return (
            "Parse XML and extract tags, attributes, and text. "
            "Supports XPath queries, XML→dict conversion, and validation. "
            "Works with SOAP APIs, Atom/RSS feeds, config files, and any XML payload. "
            "All stdlib — no external dependencies."
        )

    def definitions(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "xml_extract",
                "description": (
                    "Return the text content of all elements with a given tag name. "
                    "E.g. xml_extract(xml, 'title') returns all title text values."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "xml_string": {"type": "string", "description": "XML document string"},
                        "tag": {"type": "string", "description": "Tag name to extract (without namespace)"},
                    },
                    "required": ["xml_string", "tag"],
                },
            },
            {
                "name": "xml_attrs",
                "description": (
                    "Return attributes of all elements with a given tag name as a list of dicts."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "xml_string": {"type": "string", "description": "XML document string"},
                        "tag": {"type": "string", "description": "Tag name to match"},
                    },
                    "required": ["xml_string", "tag"],
                },
            },
            {
                "name": "xml_find",
                "description": (
                    "Find the first element matching an XPath expression. "
                    "Returns tag, text, attrs, and child tag names. "
                    "Example xpath: './/item', './/{http://…}title', './/item[@id=\"2\"]'"
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "xml_string": {"type": "string", "description": "XML document string"},
                        "xpath": {"type": "string", "description": "XPath expression"},
                    },
                    "required": ["xml_string", "xpath"],
                },
            },
            {
                "name": "xml_findall",
                "description": (
                    "Return all elements matching an XPath expression. "
                    "Each result has tag, text, and attrs."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "xml_string": {"type": "string", "description": "XML document string"},
                        "xpath": {"type": "string", "description": "XPath expression"},
                    },
                    "required": ["xml_string", "xpath"],
                },
            },
            {
                "name": "xml_to_dict",
                "description": (
                    "Convert an XML document to a nested Python dict (returned as JSON). "
                    "Attributes use @ prefix, text content uses #text when mixed with children. "
                    "Repeated tags become lists."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "xml_string": {"type": "string", "description": "XML document string"},
                    },
                    "required": ["xml_string"],
                },
            },
            {
                "name": "xml_validate",
                "description": "Check whether an XML string is well-formed. Returns {valid: true/false}.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "xml_string": {"type": "string", "description": "XML document string"},
                    },
                    "required": ["xml_string"],
                },
            },
            {
                "name": "xml_tags",
                "description": (
                    "Return all unique tag names in the document with occurrence counts. "
                    "Useful for exploring an unfamiliar XML structure."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "xml_string": {"type": "string", "description": "XML document string"},
                    },
                    "required": ["xml_string"],
                },
            },
        ]

    def execute(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        if tool_name == "xml_extract":
            return self.xml_extract(**arguments)
        if tool_name == "xml_attrs":
            return self.xml_attrs(**arguments)
        if tool_name == "xml_find":
            return self.xml_find(**arguments)
        if tool_name == "xml_findall":
            return self.xml_findall(**arguments)
        if tool_name == "xml_to_dict":
            return self.xml_to_dict(**arguments)
        if tool_name == "xml_validate":
            return self.xml_validate(**arguments)
        if tool_name == "xml_tags":
            return self.xml_tags(**arguments)
        return json.dumps({"error": f"Unknown tool: {tool_name}"})
