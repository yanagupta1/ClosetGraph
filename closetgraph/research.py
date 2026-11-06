"""Research lookup: fetch background on an unfamiliar occasion, venue, or dress
code from Wikipedia (keyless, stdlib only).

The fetch is injected into the research agent as a callable so the agent logic
stays testable without network access. Wikipedia requires a descriptive
User-Agent header, so requests set one.
"""
import json
import logging
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

_OPENSEARCH_URL = "https://en.wikipedia.org/w/api.php"
_SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/"
_HTTP_TIMEOUT = 10
_USER_AGENT = "ClosetGraph/0.1 (portfolio project; outfit research)"


def _get_json(url):
    request = Request(url, headers={"User-Agent": _USER_AGENT})
    with urlopen(request, timeout=_HTTP_TIMEOUT) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_wikipedia(query: str) -> dict:
    """
    Resolve the best-matching Wikipedia page for a query and return its summary:
    {"title": str, "extract": str, "url": str|None}.

    Two keyless calls: opensearch to find the title, then the REST summary for
    the extract. Raises on no match, an empty extract, or a network/parse error
    so the caller can surface the failure without fabricating research.
    """
    params = {
        "action": "opensearch",
        "search": query,
        "limit": 1,
        "namespace": 0,
        "format": "json",
    }
    data = _get_json(f"{_OPENSEARCH_URL}?{urlencode(params)}")
    titles = data[1] if isinstance(data, list) and len(data) > 1 else []
    if not titles:
        raise ValueError(f"no Wikipedia match for {query!r}")

    title = titles[0]
    summary = _get_json(f"{_SUMMARY_URL}{quote(title)}")
    extract = summary.get("extract")
    if not extract:
        raise ValueError(f"no summary extract for {title!r}")

    content_urls = summary.get("content_urls") or {}
    url = (content_urls.get("desktop") or {}).get("page")
    return {"title": title, "extract": extract, "url": url}
