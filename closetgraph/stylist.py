import json
import os
import re
import time

from dotenv import load_dotenv
from groq import Groq, RateLimitError
from langfuse import get_client, observe

from closetgraph.models import OutfitCandidate, RankedOutfit, WardrobeItem


def _create_with_backoff(client, *, max_retries=2, **kwargs):
    """Call Groq chat completions with a bounded wait-and-retry on rate limits.
    A 429 is a transient per-minute token-cap error, so wait the suggested time
    and retry; this is distinct from a bad-parse, which is handled separately and
    not retried here."""
    for attempt in range(max_retries + 1):
        try:
            return client.chat.completions.create(**kwargs)
        except RateLimitError as error:
            if attempt == max_retries:
                raise
            match = re.search(r"try again in ([\d.]+)s", str(error))
            wait = (float(match.group(1)) + 2) if match else 20.0
            time.sleep(min(wait, 65))


STYLIST_PROMPT_TEMPLATE = """You are a professional personal stylist evaluating outfit options for a specific occasion.

Occasion: {occasion}

Weather: {weather}

Researched context for this occasion: {research}

Relevant style guidance retrieved for this occasion:
{guidance}

You are given a list of candidate outfits, each made of items the person already owns. Judge how the items work TOGETHER as one composed outfit, not each item in isolation.

Reasoning guidelines:
- Infer the formality level implied by the occasion description (e.g. "team happy hour, casual bar" is casual to smart casual; "first client meeting, conservative office" is business or professional).
- An item's formality can shift depending on how it's styled. A tank top alone is casual, but layered under a structured blazer with tailored trousers, the same tank can read as business casual. A silk blouse with dark jeans is more polished than the jeans alone would suggest.
- Layering pieces can raise the effective formality of what's underneath them when the combination makes sense. Penalize combinations that clash in formality or don't make practical sense together.
- Favor outfits that are well-suited and coherent over outfits that are merely acceptable.
- Weather, when provided, INFLUENCES but does NOT override the occasion. Keep the occasion's required formality, and within that, let weather shift which suitable options rank higher (e.g. for a cold, rainy client meeting still stay professional, but favor options with appropriate outerwear). Weight weather by the occasion: it matters more for outdoor or prolonged-exposure settings than for brief indoor ones. If weather is "none provided", rank without weather considerations.
- Researched context, when provided, clarifies an unfamiliar occasion or dress code. Use it to set the expected formality and constraints, and briefly explain how it shaped your interpretation. It informs the ranking but does not override owned-wardrobe limits. If it is "none", rely on the occasion as given.
Even when no candidate fully satisfies the occasion, always return your best 5 available candidates ranked by how close they come, with reasoning that explicitly names what's missing. Never return an outfit with empty items.

Candidate outfits:
{candidates}

Return your top 5 ranked outfits as a JSON array, best (1) to fifth-best (5), and nothing else, no markdown, no prose before or after. Each outfit's reasoning must describe only the items in that outfit's "items" list; do not mention any garment that is not in that outfit. Each entry must match this exact shape:

[
  {{"rank": 1, "items": ["<item_id>", "<item_id>"], "reasoning": "<one sentence on why this works for the occasion>"}},
  ...
]
"""


def _format_candidates(
    candidates: list[OutfitCandidate], wardrobe: list[WardrobeItem]
) -> str:
    """
    One compact line per candidate outfit, e.g.
    "1. t1 (black scoop-neck tank top, top) + b1 (black tailored trousers, bottom)".
    Keeps the item id (which the stylist must return), name, and category, but
    drops the verbose indented JSON - the candidate block was ~58% of the prompt
    and the repeated keys/braces were pure overhead.
    """
    wardrobe_by_id = {item.id: item for item in wardrobe}
    lines = []
    for index, candidate in enumerate(candidates, start=1):
        parts = [
            f"{item_id} ({wardrobe_by_id[item_id].name}, {wardrobe_by_id[item_id].category})"
            for item_id in candidate.items
        ]
        lines.append(f"{index}. " + " + ".join(parts))
    return "\n".join(lines)


def _extract_json_array(content: str | None) -> str:
    if not content or not content.strip():
        raise ValueError("Model response was empty.")

    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text).strip()

    if text.startswith("{"):
        obj = json.loads(text)
        for key in ("ranked_outfits", "outfits", "results", "rankings"):
            if key in obj:
                return json.dumps(obj[key])

    if text.startswith("["):
        return text

    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]

    raise ValueError("Model response did not contain a JSON array.")


def _parse_ranked_outfits(content: str) -> list[RankedOutfit]:
    data = json.loads(_extract_json_array(content))
    if not isinstance(data, list):
        raise ValueError("Expected a JSON array of ranked outfits.")

    ranked = [RankedOutfit(**item) for item in data]
    return sorted(ranked, key=lambda outfit: outfit.rank)


def _format_weather(weather_context) -> str:
    """Human-readable weather line for the prompt, or a clear 'none' marker."""
    if weather_context is None:
        return "none provided"
    parts = []
    if weather_context.temperature_c is not None:
        parts.append(f"{weather_context.temperature_c:g} degrees celsius")
    if weather_context.condition:
        parts.append(weather_context.condition)
    return ", ".join(parts) or "none provided"


def _format_research(research_context) -> str:
    """Human-readable research line for the prompt, or a clear 'none' marker."""
    if research_context is None:
        return "none"
    parts = [research_context.summary]
    if research_context.inferred_formality:
        parts.append(f"(expected formality: {research_context.inferred_formality})")
    return " ".join(parts)


@observe()
def rank_outfits(
    candidates: list[OutfitCandidate],
    wardrobe: list[WardrobeItem],
    occasion: str,
    retrieved_guidance: list[str] | None = None,
    weather_context=None,
    research_context=None,
) -> list[RankedOutfit]:
    load_dotenv()
    model = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

    guidance_text = "\n\n".join(retrieved_guidance or []) or "(none retrieved)"
    prompt = STYLIST_PROMPT_TEMPLATE.format(
        occasion=occasion,
        weather=_format_weather(weather_context),
        research=_format_research(research_context),
        guidance=guidance_text,
        candidates=_format_candidates(candidates, wardrobe),
    )

    client = Groq()
    langfuse = get_client()
    # Reads LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY from env (loaded above
    # via load_dotenv). If unset, the client is disabled and these calls no-op.
    langfuse.update_current_span(
        input={"occasion": occasion},
        metadata={
            "num_candidates": len(candidates),
            "num_retrieved_guidance": len(retrieved_guidance or []),
        },
    )

    with langfuse.start_as_current_observation(
        name="groq-chat-completion",
        as_type="generation",
        model=model,
        input=prompt,
    ) as generation:
        response = _create_with_backoff(
            client,
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_completion_tokens=5000,
        )
        content = response.choices[0].message.content
        usage = getattr(response, "usage", None)
        if usage is not None:
            generation.update(
                output=content,
                usage_details={
                    "input": usage.prompt_tokens,
                    "output": usage.completion_tokens,
                },
            )
        else:
            generation.update(output=content)

    try:
        return _parse_ranked_outfits(content)
    except Exception as first_error:
        retry_response = _create_with_backoff(
            client,
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You return only valid JSON arrays. No markdown.",
                },
                {
                    "role": "user",
                    "content": (
                        f"{prompt}\n\nThe previous response was not valid JSON "
                        f"because: {first_error}. Return ONLY the JSON array."
                    ),
                },
            ],
            temperature=0,
            max_completion_tokens=5000,
        )
        retry_content = retry_response.choices[0].message.content
        return _parse_ranked_outfits(retry_content)
