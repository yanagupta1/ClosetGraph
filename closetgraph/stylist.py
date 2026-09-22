import json
import os
import re

from dotenv import load_dotenv
from groq import Groq

from closetgraph.models import OutfitCandidate, RankedOutfit, WardrobeItem


STYLIST_PROMPT_TEMPLATE = """You are a professional personal stylist evaluating outfit options for a specific occasion.

Occasion: {occasion}

You are given a list of candidate outfits, each made of items the person already owns. Judge how the items work TOGETHER as one composed outfit, not each item in isolation.

Reasoning guidelines:
- Infer the formality level implied by the occasion description (e.g. "team happy hour, casual bar" is casual to smart casual; "first client meeting, conservative office" is business or professional).
- An item's formality can shift depending on how it's styled. A tank top alone is casual, but layered under a structured blazer with tailored trousers, the same tank can read as business casual. A silk blouse with dark jeans is more polished than the jeans alone would suggest.
- Layering pieces can raise the effective formality of what's underneath them when the combination makes sense. Penalize combinations that clash in formality or don't make practical sense together.
- Favor outfits that are well-suited and coherent over outfits that are merely acceptable.

Candidate outfits:
{candidates}

Return your top 5 ranked outfits as a JSON array, best (1) to fifth-best (5), and nothing else, no markdown, no prose before or after. Each entry must match this exact shape:

[
  {{"rank": 1, "items": ["<item_id>", "<item_id>"], "reasoning": "<one sentence on why this works for the occasion>"}},
  ...
]
"""


def _resolve_candidates(
    candidates: list[OutfitCandidate], wardrobe: list[WardrobeItem]
) -> list[dict[str, object]]:
    wardrobe_by_id = {item.id: item for item in wardrobe}
    resolved = []

    for candidate in candidates:
        resolved.append(
            {
                "items": [
                    wardrobe_by_id[item_id].model_dump()
                    for item_id in candidate.items
                ]
            }
        )

    return resolved


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


def _fallback_rank_outfits(
    candidates: list[OutfitCandidate],
    wardrobe: list[WardrobeItem],
    occasion: str,
) -> list[RankedOutfit]:
    wardrobe_by_id = {item.id: item for item in wardrobe}
    occasion_lower = occasion.lower()

    professional_terms = {
        "client",
        "meeting",
        "office",
        "conservative",
        "interview",
        "professional",
        "business",
        "formal",
    }
    casual_terms = {"casual", "bar", "happy hour", "weekend", "relaxed"}
    wants_professional = any(term in occasion_lower for term in professional_terms)
    wants_casual = any(term in occasion_lower for term in casual_terms)

    polished_words = {
        "blazer",
        "structured",
        "tailored",
        "trousers",
        "silk",
        "button-down",
        "turtleneck",
        "pleated",
        "midi",
        "fitted",
        "wide-leg",
        "blouse",
    }
    casual_words = {"tank", "tee", "graphic", "denim", "jeans", "oversized"}
    soft_words = {"cardigan", "knit", "cream", "beige", "camel"}

    def score(candidate: OutfitCandidate) -> tuple[int, int, str]:
        names = [wardrobe_by_id[item_id].name.lower() for item_id in candidate.items]
        text = " ".join(names)
        value = 0
        value += sum(3 for word in polished_words if word in text)
        value -= sum(2 for word in casual_words if word in text)
        value += sum(1 for word in soft_words if word in text)
        if any("blazer" in name for name in names):
            value += 4 if wants_professional else 1
        if any("tailored trousers" in name for name in names):
            value += 3
        if any("silk" in name or "button-down" in name for name in names):
            value += 3
        if any("jeans" in name or "denim" in name for name in names):
            value += 1 if wants_casual else -3
        if wants_professional and len(candidate.items) == 3:
            value += 2
        return (value, len(candidate.items), text)

    ranked_candidates = sorted(candidates, key=score, reverse=True)[:5]
    results: list[RankedOutfit] = []
    for index, candidate in enumerate(ranked_candidates, start=1):
        if wants_professional:
            reasoning = (
                "This reads polished and office-appropriate because the pieces "
                "combine structured or refined elements for the occasion."
            )
        elif wants_casual:
            reasoning = (
                "This balances comfort and coherence while still looking put "
                "together for the casual setting."
            )
        else:
            reasoning = (
                "This is a coherent combination using owned pieces that suit "
                "the stated occasion."
            )
        results.append(
            RankedOutfit(rank=index, items=candidate.items, reasoning=reasoning)
        )
    return results


def rank_outfits(
    candidates: list[OutfitCandidate],
    wardrobe: list[WardrobeItem],
    occasion: str,
) -> list[RankedOutfit]:
    load_dotenv()
    model = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

    resolved_candidates = _resolve_candidates(candidates, wardrobe)
    prompt = STYLIST_PROMPT_TEMPLATE.format(
        occasion=occasion,
        candidates=json.dumps(resolved_candidates, indent=2),
    )

    try:
        client = Groq()
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        content = response.choices[0].message.content

        try:
            return _parse_ranked_outfits(content)
        except Exception as first_error:
            retry_response = client.chat.completions.create(
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
            )
            retry_content = retry_response.choices[0].message.content
            return _parse_ranked_outfits(retry_content)
    except Exception:
        return _fallback_rank_outfits(candidates, wardrobe, occasion)
