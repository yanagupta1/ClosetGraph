"""Supervisor: decide whether an occasion needs external research.

Hybrid routing. A deterministic fast-path skips research (no LLM call) only when
the occasion clearly matches a dress code our style corpus already covers AND
shows no other sign of an unfamiliar venue or event. Everything else falls
through to a cheap LLM classifier. Deciding relevance here is a knowledge-gap
judgment, which is why an LLM is justified in this role (unlike weather gating).
"""
import json
import logging
import os

from dotenv import load_dotenv
from groq import Groq
from langfuse import observe

from closetgraph.research import _extract_json_object

logger = logging.getLogger(__name__)

# Dress codes / occasion types the style corpus covers; a plain match here is a
# candidate for the deterministic skip.
_COVERED_DRESS_CODES = (
    "casual",
    "smart casual",
    "business casual",
    "business",
    "professional",
    "office",
    "interview",
    "client meeting",
    "cocktail",
    "semi-formal",
    "formal",
    "black tie",
    "black-tie",
    "brunch",
    "lunch",
    "dinner",
    "date",
    "errands",
    "happy hour",
    "bar",
    "nightclub",
)

# Signals that something specific/unfamiliar may be involved, so do not skip even
# if a familiar dress code also appears (e.g. "cocktail party at the Met Gala").
_ESCALATION_MARKERS = (
    "gala",
    "ball",
    "masquerade",
    "themed",
    "theme party",
    "white tie",
    "white-tie",
    "regatta",
    "derby",
    "debutante",
    "quinceanera",
    "quinceañera",
    "mitzvah",
    "premiere",
    "benefit",
    "garden party",
    "costume",
)

SUPERVISOR_PROMPT_TEMPLATE = """You decide whether an outfit request needs a quick external lookup about its attire. Research is needed ONLY when external information would resolve a CONCRETE attire uncertainty that MATERIALLY changes what to wear.

Do NOT research just because an occasion sounds unfamiliar, cultural, themed, or activity-specific. Guest attire for a quinceanera, a Dia de los Muertos celebration, a masquerade ball, a Renaissance faire, or spectating a polo match is broadly inferable from general knowledge, so these do NOT need research. An unfamiliar venue NAME alone is never a reason (a cafe or park has no special attire even if unknown).

DO research when the occasion names or states a specific dress code whose meaning is not general knowledge, or a venue/event with a particular published code you cannot determine. Examples that DO need research: a stated 'beach formal', 'smart elegant', or 'club rig' code; a members-only club's stated-but-unspecified code; the Royal Enclosure at Royal Ascot; the Stewards' Enclosure at Henley.

Common occasions (office, dinner, casual, cocktail, black tie) do NOT need research, their attire is already well understood.

Occasion: {occasion}

Return ONLY a single JSON object, no markdown:
{{"research_needed": true or false, "research_query": "<a concise searchable term, the dress code or event name alone (for example \"cocktail attire\", \"white tie\", \"Royal Ascot\"), not a full sentence, or null>", "research_reason": "<one short clause: what specific attire is uncertain and why, or why no research is needed>"}}
"""


def _has_research_signal(occasion: str) -> bool:
    """Signals that only DEFER the decision to the classifier, never force
    research: an explicit stated dress code, an escalation marker, or a
    capitalized named venue/event. The classifier then judges whether the attire
    is actually inferable."""
    lower = occasion.lower()
    # A stated code is a strong defer signal: a covered word like "dinner" must
    # not short-circuit "dinner ... with a dress code".
    if "dress code" in lower or "attire" in lower:
        return True
    if any(marker in lower for marker in _ESCALATION_MARKERS):
        return True
    for word in occasion.split()[1:]:
        if word[:1].isupper():
            return True
    return False


def _llm_classify(occasion: str) -> dict:
    load_dotenv()
    model = os.getenv("SUPERVISOR_MODEL", os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"))
    try:
        response = Groq().chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": SUPERVISOR_PROMPT_TEMPLATE.format(occasion=occasion)}
            ],
            temperature=0,
            # Low reasoning effort already keeps this well under 1000 tokens, so a
            # small completion cap is enough and sharply cuts the TPM reservation
            # (prompt + cap) when this runs back-to-back with research and stylist
            # in one request. The previous benchmark (28/29) used a 5000 cap; the
            # lower cap is re-checked on the hard routing cases.
            max_completion_tokens=1024,
            reasoning_effort="low",
        )
        data = json.loads(_extract_json_object(response.choices[0].message.content))
    except Exception as error:  # fail safe: on classifier failure, do not research
        logger.warning("supervisor classifier failed for %r: %r", occasion, error)
        return {
            "research_needed": False,
            "research_query": None,
            "research_reason": "classifier unavailable; proceeding without research",
        }
    needed = bool(data.get("research_needed"))
    query = data.get("research_query") if needed else None
    if query in ("null", ""):
        query = None
    return {
        "research_needed": needed,
        "research_query": query,
        "research_reason": data.get("research_reason"),
    }


@observe()
def supervise(occasion: str, classifier=None) -> dict:
    """
    Decide research routing for an occasion, returning
    {"research_needed": bool, "research_query": str|None, "research_reason": str|None}.

    Deterministic skip when the occasion matches a covered dress code and shows
    no deferral signal; otherwise the classifier judges whether the dress code is
    actually inferable (an unfamiliar venue name alone is not a reason to research).
    """
    lower = occasion.lower()
    covered = any(dc in lower for dc in _COVERED_DRESS_CODES)
    if covered and not _has_research_signal(occasion):
        return {
            "research_needed": False,
            "research_query": None,
            "research_reason": "familiar occasion; dress code inferable from general knowledge",
        }
    return (classifier or _llm_classify)(occasion)
