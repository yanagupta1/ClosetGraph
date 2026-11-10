"""Research lookup: fetch background on an unfamiliar occasion, venue, or dress
code from Wikipedia (keyless, stdlib only).

The fetch is injected into the research agent as a callable so the agent logic
stays testable without network access. Wikipedia requires a descriptive
User-Agent header, so requests set one.
"""
import json
import logging
import os
import re
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from dotenv import load_dotenv
from groq import Groq
from langfuse import observe

from closetgraph.models import ResearchContext

logger = logging.getLogger(__name__)

_WIKI_API_URL = "https://en.wikipedia.org/w/api.php"
_HTTP_TIMEOUT = 10
_USER_AGENT = "ClosetGraph/0.1 (portfolio project; outfit research)"

# Neutral extractor: distill a source into styling constraints. The styling
# judgment stays with the stylist; this only summarizes retrieved facts.
RESEARCH_PROMPT_TEMPLATE = """You are a styling research assistant. From the background text about an occasion, venue, or dress code, extract only styling-relevant constraints. Do not invent facts beyond the text.

Query: {query}

Source extract:
{extract}

Return ONLY a single JSON object, no markdown and no prose, matching exactly:
{{"summary": "<one or two sentences of styling-relevant constraints from the text>", "inferred_formality": "<short label such as 'formal / black-tie', or null if the text does not make it clear>"}}
"""


def _get_json(url):
    request = Request(url, headers={"User-Agent": _USER_AGENT})
    with urlopen(request, timeout=_HTTP_TIMEOUT) as response:
        return json.loads(response.read().decode("utf-8"))


# Qualifier words that rarely appear in page titles; stripping them turns a
# verbose query like "Royal Enclosure dress code at Royal Ascot" into something
# opensearch can match.
_QUALIFIER_RE = re.compile(
    r"\b(dress[- ]?code|attire|requirements?|what to wear|rules?|policy)\b", re.IGNORECASE
)


def _query_variants(query: str):
    variants = []

    def add(candidate):
        candidate = " ".join(candidate.split())
        if candidate and candidate.lower() not in [v.lower() for v in variants]:
            variants.append(candidate)

    add(query)
    stripped = _QUALIFIER_RE.sub(" ", query)
    add(stripped)
    add(re.sub(r"^\s*(the|a|an)\s+", "", stripped, flags=re.IGNORECASE))
    return variants


def _resolve_title(query: str):
    params = {"action": "opensearch", "search": query, "limit": 1, "namespace": 0, "format": "json"}
    data = _get_json(f"{_WIKI_API_URL}?{urlencode(params)}")
    titles = data[1] if isinstance(data, list) and len(data) > 1 else []
    return titles[0] if titles else None


def _fetch_page(title: str):
    """Fetch a page's full plain-text extract and canonical URL in one keyless
    query call (prop=extracts gives all sections, not just the lead)."""
    params = {
        "action": "query",
        "prop": "extracts|info",
        "explaintext": 1,
        "inprop": "url",
        "redirects": 1,
        "titles": title,
        "format": "json",
    }
    data = _get_json(f"{_WIKI_API_URL}?{urlencode(params)}")
    pages = (data.get("query") or {}).get("pages") or {}
    for page in pages.values():
        return page.get("extract") or "", page.get("fullurl")
    return "", None


# Attire markers used to locate a dress-code passage anywhere in the article, so
# a code in a later section (not the lead) is still captured.
_ATTIRE_KEYWORDS = ("dress code", "dress-code", "attire", "what to wear", "white tie", "black tie")


def _relevant_passage(text: str, max_chars: int = 1500) -> str:
    lower = text.lower()
    hits = [lower.find(kw) for kw in _ATTIRE_KEYWORDS if lower.find(kw) != -1]
    if not hits:
        return ""
    start = max(0, min(hits) - 250)
    return text[start : start + max_chars].strip()


def fetch_wikipedia(query: str) -> dict:
    """
    Resolve the best-matching Wikipedia page for a query and return a passage:
    {"title": str, "extract": str, "url": str|None}.

    Keyless: opensearch to find the title (trying the query, then simplified
    variants so a verbose "X dress code at Y" still resolves), then a single
    query call for the full plain-text extract. The returned extract is the
    passage around the first attire/dress-code mention when the article has one
    (so a code in a later section is captured), else the lead. Raises on no
    match, empty content, or a network/parse error so the caller can surface the
    failure without fabricating research.
    """
    title = None
    for variant in _query_variants(query):
        title = _resolve_title(variant)
        if title:
            break
    if not title:
        raise ValueError(f"no Wikipedia match for {query!r}")

    full_text, url = _fetch_page(title)
    if not full_text:
        raise ValueError(f"no content for {title!r}")

    extract = _relevant_passage(full_text) or full_text[:800].strip()
    return {"title": title, "extract": extract, "url": url}


def _extract_json_object(content: str | None) -> str:
    if not content or not content.strip():
        raise ValueError("Research summary response was empty.")
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Research summary response did not contain a JSON object.")
    return text[start : end + 1]


@observe()
def run_research(query: str, fetcher=fetch_wikipedia):
    """
    Run the research agent for a query: fetch background, then distill it into a
    ResearchContext with one LLM call. Returns (context, error). On a lookup or
    summary failure, returns (None, "<reason>") without fabricating research, so
    the caller can continue and surface the error.
    """
    load_dotenv()
    model = os.getenv("RESEARCH_MODEL", os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"))

    try:
        fetched = fetcher(query)
    except Exception as error:
        message = f"research lookup failed for {query!r}: {error!r}"
        logger.warning(message)
        return None, message

    prompt = RESEARCH_PROMPT_TEMPLATE.format(
        query=query, extract=fetched["extract"][:2000]
    )
    try:
        response = Groq().chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_completion_tokens=2000,
        )
        data = json.loads(_extract_json_object(response.choices[0].message.content))
    except Exception as error:
        message = f"research summary failed for {query!r}: {error!r}"
        logger.warning(message)
        return None, message

    inferred = data.get("inferred_formality")
    return (
        ResearchContext(
            query=query,
            summary=data["summary"],
            inferred_formality=inferred if inferred not in (None, "null", "") else None,
            source_title=fetched["title"],
            source_url=fetched["url"],
        ),
        None,
    )
