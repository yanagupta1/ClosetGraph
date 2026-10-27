import json
import os
import re

from dotenv import load_dotenv
from groq import Groq
from langfuse import get_client, observe

from closetgraph.models import OutfitCandidate, RankedOutfit, WardrobeItem


STYLIST_PROMPT_TEMPLATE = """You are a professional personal stylist evaluating outfit options for a specific occasion.

Occasion: {occasion}

Relevant style guidance retrieved for this occasion:
{guidance}

You are given a list of candidate outfits, each made of items the person already owns. Judge how the items work TOGETHER as one composed outfit, not each item in isolation.

Reasoning guidelines:
- Infer the formality level implied by the occasion description (e.g. "team happy hour, casual bar" is casual to smart casual; "first client meeting, conservative office" is business or professional).
- An item's formality can shift depending on how it's styled. A tank top alone is casual, but layered under a structured blazer with tailored trousers, the same tank can read as business casual. A silk blouse with dark jeans is more polished than the jeans alone would suggest.
- Layering pieces can raise the effective formality of what's underneath them when the combination makes sense. Penalize combinations that clash in formality or don't make practical sense together.
- Favor outfits that are well-suited and coherent over outfits that are merely acceptable.
Even when no candidate fully satisfies the occasion, always return your best 5 available candidates ranked by how close they come, with reasoning that explicitly names what's missing. Never return an outfit with empty items.

Candidate outfits:
{candidates}

Return your top 5 ranked outfits as a JSON array, best (1) to fifth-best (5), and nothing else, no markdown, no prose before or after. Each outfit's reasoning must describe only the items in that outfit's "items" list; do not mention any garment that is not in that outfit. Each entry must match this exact shape:

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


@observe()
def rank_outfits(
    candidates: list[OutfitCandidate],
    wardrobe: list[WardrobeItem],
    occasion: str,
    retrieved_guidance: list[str] | None = None,
) -> list[RankedOutfit]:
    load_dotenv()
    model = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

    guidance_text = "\n\n".join(retrieved_guidance or []) or "(none retrieved)"
    resolved_candidates = _resolve_candidates(candidates, wardrobe)
    prompt = STYLIST_PROMPT_TEMPLATE.format(
        occasion=occasion,
        guidance=guidance_text,
        candidates=json.dumps(resolved_candidates, indent=2),
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
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_completion_tokens=8192,
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
            max_completion_tokens=8192,
        )
        retry_content = retry_response.choices[0].message.content
        return _parse_ranked_outfits(retry_content)
