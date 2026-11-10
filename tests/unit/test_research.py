import json

import pytest

from closetgraph.models import ResearchContext
from closetgraph.research import _extract_json_object, fetch_wikipedia, run_research


class FakeMessage:
    def __init__(self, content):
        self.content = content


class FakeChoice:
    def __init__(self, content):
        self.message = FakeMessage(content)


class FakeResponse:
    def __init__(self, content):
        self.choices = [FakeChoice(content)]


class FakeCompletions:
    def __init__(self, content):
        self.content = content
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return FakeResponse(self.content)


class FakeChat:
    def __init__(self, completions):
        self.completions = completions


class FakeGroq:
    completions = None

    def __init__(self):
        self.chat = FakeChat(self.completions)


class FakeHTTPResponse:
    def __init__(self, payload):
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return json.dumps(self._payload).encode("utf-8")


def _page(extract, url="https://en.wikipedia.org/wiki/Example"):
    return {"query": {"pages": {"1": {"extract": extract, "fullurl": url}}}}


def _fake_urlopen(opensearch=None, page=None, raise_on=None):
    """urlopen stub returning opensearch vs full-extract (action=query) payloads."""

    def urlopen(request, timeout=None):
        url = getattr(request, "full_url", request)
        if raise_on and raise_on in url:
            raise OSError("network down")
        if "action=query" in url:
            return FakeHTTPResponse(page)
        return FakeHTTPResponse(opensearch)

    return urlopen


def test_fetch_wikipedia_success(monkeypatch):
    monkeypatch.setattr(
        "closetgraph.research.urlopen",
        _fake_urlopen(
            opensearch=["black tie", ["Black tie"], [""], ["https://en.wikipedia.org/wiki/Black_tie"]],
            page=_page(
                "Black tie is a semi-formal Western dress code for evening events.",
                "https://en.wikipedia.org/wiki/Black_tie",
            ),
        ),
    )

    result = fetch_wikipedia("black tie")

    assert result["title"] == "Black tie"
    assert "semi-formal" in result["extract"]
    assert result["url"].endswith("Black_tie")


def test_fetch_wikipedia_extracts_dress_code_passage_from_later_section(monkeypatch):
    # Attire appears deep in the article (not the lead); the passage around the
    # "dress code" mention should be returned, not the irrelevant lead.
    lead = "Royal Ascot is a famous horse racing event. " * 20
    body = "History section with lots of unrelated text. " * 20
    dress = "Dress code: the Royal Enclosure requires morning dress and a top hat."
    monkeypatch.setattr(
        "closetgraph.research.urlopen",
        _fake_urlopen(
            opensearch=["Royal Ascot", ["Royal Ascot"], [""], [""]],
            page=_page(lead + body + dress, "https://en.wikipedia.org/wiki/Royal_Ascot"),
        ),
    )

    result = fetch_wikipedia("Royal Ascot")

    assert "morning dress" in result["extract"]
    assert "top hat" in result["extract"]


def test_fetch_wikipedia_falls_back_to_simplified_query(monkeypatch):
    # The verbose query has no match; the simplified variant (qualifier stripped)
    # does. fetch_wikipedia should try the variant and succeed.
    def urlopen(request, timeout=None):
        url = getattr(request, "full_url", request)
        if "action=query" in url:
            return FakeHTTPResponse(_page("Royal Ascot is a famous horse racing event."))
        from urllib.parse import unquote

        matched = "dress" not in unquote(url).lower()
        return FakeHTTPResponse(["q", (["Royal Ascot"] if matched else []), [""], [""]])

    monkeypatch.setattr("closetgraph.research.urlopen", urlopen)

    result = fetch_wikipedia("Royal Ascot dress code")

    assert result["title"] == "Royal Ascot"


def test_fetch_wikipedia_empty_pages_raises(monkeypatch):
    # opensearch resolves a title, but the extract query returns no pages at all
    # -> _fetch_page yields ("", None) and fetch_wikipedia reports no content.
    monkeypatch.setattr(
        "closetgraph.research.urlopen",
        _fake_urlopen(
            opensearch=["q", ["Some Page"], [""], ["u"]],
            page={"query": {"pages": {}}},
        ),
    )

    with pytest.raises(ValueError, match="no content"):
        fetch_wikipedia("q")


def test_extract_json_object_raises_on_empty():
    with pytest.raises(ValueError, match="empty"):
        _extract_json_object("")


def test_extract_json_object_strips_markdown_fence():
    assert _extract_json_object('```json\n{"summary": "x"}\n```') == '{"summary": "x"}'


def test_fetch_wikipedia_no_match_raises(monkeypatch):
    monkeypatch.setattr(
        "closetgraph.research.urlopen",
        _fake_urlopen(opensearch=["xyzzy", [], [], []]),
    )

    with pytest.raises(ValueError, match="no Wikipedia match"):
        fetch_wikipedia("xyzzy no such page")


def test_fetch_wikipedia_empty_extract_raises(monkeypatch):
    monkeypatch.setattr(
        "closetgraph.research.urlopen",
        _fake_urlopen(
            opensearch=["q", ["Some Page"], [""], ["u"]],
            page=_page("", "u"),
        ),
    )

    with pytest.raises(ValueError, match="no content"):
        fetch_wikipedia("q")


def test_fetch_wikipedia_network_error_propagates(monkeypatch):
    monkeypatch.setattr(
        "closetgraph.research.urlopen", _fake_urlopen(raise_on="api.php")
    )

    with pytest.raises(OSError):
        fetch_wikipedia("black tie")


# --- run_research (fetcher + Groq both mocked) ---

def _fake_fetch(query):
    return {
        "title": "Black tie",
        "extract": "Black tie is a semi-formal evening dress code.",
        "url": "https://en.wikipedia.org/wiki/Black_tie",
    }


def test_run_research_success(monkeypatch):
    FakeGroq.completions = FakeCompletions(
        '{"summary": "Formal evening wear; dark suit or gown.", "inferred_formality": "formal / black-tie"}'
    )
    monkeypatch.setattr("closetgraph.research.Groq", FakeGroq)

    context, error = run_research("black tie", fetcher=_fake_fetch)

    assert error is None
    assert isinstance(context, ResearchContext)
    assert context.inferred_formality == "formal / black-tie"
    assert context.source_url.endswith("Black_tie")
    assert "evening" in context.summary.lower()


def test_run_research_lookup_failure_returns_error(monkeypatch):
    def boom(query):
        raise OSError("network down")

    monkeypatch.setattr("closetgraph.research.Groq", FakeGroq)

    context, error = run_research("black tie", fetcher=boom)

    assert context is None  # no fabricated research
    assert "network down" in error


def test_run_research_bad_summary_returns_error(monkeypatch):
    FakeGroq.completions = FakeCompletions("not json at all")
    monkeypatch.setattr("closetgraph.research.Groq", FakeGroq)

    context, error = run_research("black tie", fetcher=_fake_fetch)

    assert context is None
    assert "research summary failed" in error


def test_run_research_normalizes_null_formality(monkeypatch):
    FakeGroq.completions = FakeCompletions(
        '{"summary": "Unclear dress code.", "inferred_formality": "null"}'
    )
    monkeypatch.setattr("closetgraph.research.Groq", FakeGroq)

    context, error = run_research("obscure event", fetcher=_fake_fetch)

    assert context.inferred_formality is None
