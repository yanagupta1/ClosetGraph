import json

import pytest

from closetgraph.models import ResearchContext
from closetgraph.research import fetch_wikipedia, run_research


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


def _fake_urlopen(opensearch=None, summary=None, raise_on=None):
    """urlopen stub returning opensearch vs REST-summary payloads by URL."""

    def urlopen(request, timeout=None):
        url = getattr(request, "full_url", request)
        if raise_on and raise_on in url:
            raise OSError("network down")
        if "api/rest_v1/page/summary" in url:
            return FakeHTTPResponse(summary)
        return FakeHTTPResponse(opensearch)

    return urlopen


def test_fetch_wikipedia_success(monkeypatch):
    monkeypatch.setattr(
        "closetgraph.research.urlopen",
        _fake_urlopen(
            opensearch=["black tie", ["Black tie"], [""], ["https://en.wikipedia.org/wiki/Black_tie"]],
            summary={
                "extract": "Black tie is a semi-formal Western dress code for evening events.",
                "content_urls": {"desktop": {"page": "https://en.wikipedia.org/wiki/Black_tie"}},
            },
        ),
    )

    result = fetch_wikipedia("black tie")

    assert result["title"] == "Black tie"
    assert "semi-formal" in result["extract"]
    assert result["url"].endswith("Black_tie")


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
            summary={"extract": ""},
        ),
    )

    with pytest.raises(ValueError, match="no summary extract"):
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
