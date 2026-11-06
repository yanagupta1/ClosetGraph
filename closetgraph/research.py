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
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from dotenv import load_dotenv
from groq import Groq
from langfuse import observe

from closetgraph.models import ResearchContext

logger = logging.getLogger(__name__)

_OPENSEARCH_URL = "https://en.wikipedia.org/w/api.php"
_SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/"
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
