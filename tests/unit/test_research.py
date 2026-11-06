import json

import pytest

from closetgraph.research import fetch_wikipedia


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
