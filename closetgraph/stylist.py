import json

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
                    wardrobe_by_id[item_id].dict()
                    for item_id in candidate.items
                ]
            }
        )

    return resolved


def _parse_ranked_outfits(content: str) -> list[RankedOutfit]:
    data = json.loads(content)
    return [RankedOutfit(**item) for item in data]


def rank_outfits(
    candidates: list[OutfitCandidate],
    wardrobe: list[WardrobeItem],
    occasion: str,
) -> list[RankedOutfit]:
    load_dotenv()
    client = Groq()

    resolved_candidates = _resolve_candidates(candidates, wardrobe)
    prompt = STYLIST_PROMPT_TEMPLATE.format(
        occasion=occasion,
        candidates=json.dumps(resolved_candidates, indent=2),
    )

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
    )
    content = response.choices[0].message.content

    try:
        return _parse_ranked_outfits(content)
    except Exception as first_error:
        retry_response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "user",
                    "content": f"{prompt}\n\nReturn ONLY valid JSON, no other text.",
                }
            ],
        )
        retry_content = retry_response.choices[0].message.content

        try:
            return _parse_ranked_outfits(retry_content)
        except Exception as second_error:
            raise ValueError(
                "Failed to parse Groq response as list[RankedOutfit] after one retry."
            ) from second_error
