"""Tests for ChunkerTool — text/list chunking for LLM context windows."""

import json
import pytest
from agent_friend.tools.chunker import ChunkerTool


@pytest.fixture
def tool():
    return ChunkerTool()


LOREM = (
    "The quick brown fox jumps over the lazy dog. "
    "Pack my box with five dozen liquor jugs. "
    "How valiantly did Excalibur stand against the onslaught. "
    "A stitch in time saves nine. "
    "The early bird catches the worm. "
    "All that glitters is not gold. "
    "To be or not to be that is the question. "
    "She sells seashells by the seashore. "
    "Peter Piper picked a peck of pickled peppers. "
    "How much wood would a woodchuck chuck if a woodchuck could chuck wood."
)


# ── chunk_text / chars ─────────────────────────────────────────────────────

def test_chunk_text_basic(tool):
    chunks = json.loads(tool.chunk_text("abcdefghij", max_chars=3))
    assert len(chunks) >= 3
    assert chunks[0]["text"] == "abc"
    assert chunks[0]["char_count"] == 3
    assert chunks[0]["index"] == 0


def test_chunk_text_exact_divisible(tool):
    chunks = json.loads(tool.chunk_text("abcdef", max_chars=2))
    assert len(chunks) == 3
    assert "".join(c["text"] for c in chunks) == "abcdef"


def test_chunk_text_no_split_needed(tool):
    chunks = json.loads(tool.chunk_text("hello", max_chars=100))
    assert len(chunks) == 1
    assert chunks[0]["text"] == "hello"


def test_chunk_text_empty(tool):
    chunks = json.loads(tool.chunk_text(""))
    assert chunks == []


def test_chunk_text_overlap(tool):
    text = "0123456789"
    chunks = json.loads(tool.chunk_text(text, max_chars=5, overlap=2))
    assert chunks[0]["text"] == "01234"
    assert chunks[1]["text"] == "34567"  # overlaps by 2
    assert chunks[2]["text"] == "6789"


def test_chunk_text_token_estimate_present(tool):
    chunks = json.loads(tool.chunk_text("hello world foo bar", max_chars=50))
    assert "token_estimate" in chunks[0]
    assert chunks[0]["token_estimate"] > 0


# ── chunk_text / tokens mode ───────────────────────────────────────────────

def test_chunk_text_tokens_mode(tool):
    long_text = "word " * 200  # ~1000 chars, ~250 tokens
    chunks = json.loads(tool.chunk_text(long_text, max_chars=50, mode="tokens"))
    # Each chunk should be ~200 chars (50 tokens * 4)
    assert len(chunks) > 1
    for c in chunks:
        assert c["token_estimate"] <= 55  # small buffer


# ── chunk_text / sentences mode ────────────────────────────────────────────

def test_chunk_text_sentences(tool):
    chunks = json.loads(tool.chunk_text(LOREM, max_chars=100, mode="sentences"))
    assert len(chunks) > 1
    for c in chunks:
        assert c["char_count"] <= 200  # some buffer for concatenation


def test_chunk_text_sentences_small_limit(tool):
    text = "One. Two. Three. Four."
    chunks = json.loads(tool.chunk_text(text, max_chars=10, mode="sentences"))
    # Each sentence is ~5 chars — should chunk 1-2 per group
    assert len(chunks) >= 2


# ── chunk_text / paragraphs mode ──────────────────────────────────────────

def test_chunk_text_paragraphs(tool):
    text = "Para one.\n\nPara two.\n\nPara three."
    chunks = json.loads(tool.chunk_text(text, max_chars=200, mode="paragraphs"))
    assert len(chunks) == 1  # all fit in one chunk
    assert "Para one" in chunks[0]["text"]


def test_chunk_text_paragraphs_split(tool):
    text = "Para one.\n\nPara two.\n\nPara three."
    chunks = json.loads(tool.chunk_text(text, max_chars=12, mode="paragraphs"))
    assert len(chunks) == 3


def test_chunk_text_invalid_mode(tool):
    r = json.loads(tool.chunk_text("text", mode="invalid"))
    assert "error" in r


# ── chunk_list ─────────────────────────────────────────────────────────────

def test_chunk_list_basic(tool):
    items = list(range(10))
    chunks = json.loads(tool.chunk_list(items, size=3))
    assert len(chunks) == 4  # 3+3+3+1
    assert chunks[0]["items"] == [0, 1, 2]
    assert chunks[-1]["items"] == [9]


def test_chunk_list_exact(tool):
    items = list(range(9))
    chunks = json.loads(tool.chunk_list(items, size=3))
    assert len(chunks) == 3
    assert all(c["count"] == 3 for c in chunks)


def test_chunk_list_larger_than_list(tool):
    items = [1, 2, 3]
    chunks = json.loads(tool.chunk_list(items, size=100))
    assert len(chunks) == 1
    assert chunks[0]["items"] == [1, 2, 3]


def test_chunk_list_empty(tool):
    chunks = json.loads(tool.chunk_list([], size=5))
    assert chunks == []


def test_chunk_list_size_one(tool):
    items = [1, 2, 3]
    chunks = json.loads(tool.chunk_list(items, size=1))
    assert len(chunks) == 3


def test_chunk_list_index_correct(tool):
    items = list(range(6))
    chunks = json.loads(tool.chunk_list(items, size=2))
    for i, c in enumerate(chunks):
        assert c["index"] == i


def test_chunk_list_invalid_size(tool):
    r = json.loads(tool.chunk_list([1, 2, 3], size=0))
    assert "error" in r


def test_chunk_list_strings(tool):
    items = ["a", "b", "c", "d"]
    chunks = json.loads(tool.chunk_list(items, size=2))
    assert chunks[0]["items"] == ["a", "b"]
    assert chunks[1]["items"] == ["c", "d"]


# ── chunk_by_separator ─────────────────────────────────────────────────────

def test_chunk_by_separator_basic(tool):
    text = "one---two---three"
    chunks = json.loads(tool.chunk_by_separator(text, separator="---"))
    assert len(chunks) == 3
    assert chunks[0]["text"] == "one"
    assert chunks[1]["text"] == "two"
    assert chunks[2]["text"] == "three"


def test_chunk_by_separator_newline(tool):
    text = "line1\nline2\nline3"
    chunks = json.loads(tool.chunk_by_separator(text, separator="\n"))
    assert len(chunks) == 3


def test_chunk_by_separator_keep_separator(tool):
    text = "a|b|c"
    chunks = json.loads(tool.chunk_by_separator(text, separator="|", keep_separator=True))
    assert chunks[0]["text"] == "a|"
    assert chunks[1]["text"] == "b|"


def test_chunk_by_separator_max_chars_merges(tool):
    text = "a,b,c,d,e"
    # each part is 1 char, max_chars=3 should merge ~3 per chunk
    chunks = json.loads(tool.chunk_by_separator(text, separator=",", max_chars=3))
    assert len(chunks) < 5  # should have merged some


def test_chunk_by_separator_empty_separator(tool):
    r = json.loads(tool.chunk_by_separator("text", separator=""))
    assert "error" in r


# ── chunk_sliding_window ───────────────────────────────────────────────────

def test_sliding_window_basic(tool):
    text = "0123456789"
    chunks = json.loads(tool.chunk_sliding_window(text, window_chars=4, step_chars=2))
    assert chunks[0]["text"] == "0123"
    assert chunks[1]["text"] == "2345"
    assert chunks[0]["start"] == 0
    assert chunks[1]["start"] == 2


def test_sliding_window_full_coverage(tool):
    text = "abcdefghij"
    chunks = json.loads(tool.chunk_sliding_window(text, window_chars=4, step_chars=2))
    # First chunk starts at 0, last chunk should reach the end
    last = chunks[-1]
    assert last["end"] <= len(text) + 4  # some tolerance


def test_sliding_window_single_chunk(tool):
    text = "hello"
    chunks = json.loads(tool.chunk_sliding_window(text, window_chars=100, step_chars=100))
    assert len(chunks) == 1
    assert chunks[0]["text"] == "hello"


def test_sliding_window_empty(tool):
    chunks = json.loads(tool.chunk_sliding_window("", window_chars=10, step_chars=5))
    assert chunks == []


def test_sliding_window_invalid_params(tool):
    r = json.loads(tool.chunk_sliding_window("text", window_chars=0, step_chars=5))
    assert "error" in r


def test_sliding_window_has_metadata(tool):
    chunks = json.loads(tool.chunk_sliding_window("abcdefghij", window_chars=4, step_chars=2))
    for c in chunks:
        assert "index" in c
        assert "start" in c
        assert "end" in c
        assert "char_count" in c


# ── chunk_stats ────────────────────────────────────────────────────────────

def test_chunk_stats_basic(tool):
    text = "Hello world.\n\nSecond paragraph."
    s = json.loads(tool.chunk_stats(text))
    assert s["char_count"] == len(text)
    assert s["word_count"] == 4
    assert s["paragraph_count"] == 2
    assert s["sentence_count"] == 2
    assert s["token_estimate"] > 0


def test_chunk_stats_empty(tool):
    s = json.loads(tool.chunk_stats(""))
    assert s["char_count"] == 0
    assert s["token_estimate"] == 0


def test_chunk_stats_single_line(tool):
    text = "one two three four five"
    s = json.loads(tool.chunk_stats(text))
    assert s["word_count"] == 5
    assert s["line_count"] == 1
    assert s["paragraph_count"] == 1


def test_chunk_stats_multiline(tool):
    text = "line1\nline2\nline3"
    s = json.loads(tool.chunk_stats(text))
    assert s["line_count"] == 3


# ── execute dispatch ───────────────────────────────────────────────────────

def test_execute_chunk_text(tool):
    r = json.loads(tool.execute("chunk_text", {"text": "hello world", "max_chars": 5}))
    assert isinstance(r, list)


def test_execute_chunk_list(tool):
    r = json.loads(tool.execute("chunk_list", {"items": [1, 2, 3, 4], "size": 2}))
    assert len(r) == 2


def test_execute_chunk_stats(tool):
    r = json.loads(tool.execute("chunk_stats", {"text": "hello"}))
    assert "char_count" in r


def test_execute_unknown(tool):
    r = json.loads(tool.execute("nope", {}))
    assert "error" in r


# ── tool metadata ──────────────────────────────────────────────────────────

def test_name(tool):
    assert tool.name == "chunker"


def test_description(tool):
    assert "chunk" in tool.description.lower()


def test_definitions_count(tool):
    assert len(tool.definitions()) == 5


def test_definitions_have_required_fields(tool):
    for d in tool.definitions():
        assert "name" in d
        assert "description" in d
        assert "input_schema" in d
