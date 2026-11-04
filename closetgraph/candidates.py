import random
from itertools import product

from closetgraph.models import OutfitCandidate, WardrobeItem

# Arbitrary fixed seed, for run-to-run REPRODUCIBILITY ONLY. It is deliberately
# NOT chosen to make any eval combo appear in the sample - tuning a seed to pass
# the golden set hides sampling coverage and breaks as the wardrobe grows. Eval
# checks that reference an un-sampled item are reported "inapplicable" instead
# (see eval_harness), so coverage does not depend on this value.
BASE_SAMPLE_SEED = 7


def generate_candidates(
    wardrobe: list[WardrobeItem],
    max_candidates: int = 20,
    max_accessory_variants: int = 4,
) -> list[OutfitCandidate]:
    """
    Produce a bounded candidate pool that stays small as the wardrobe grows:
    a sampled set of base outfits (top + bottom + optional layer) plus a few
    deterministic accessory variants (a base outfit with an accessory added).

    Accessories are made AVAILABLE here, not gated: whether an accessory helps
    for a given occasion or weather is the stylist's context-dependent call, not
    a rule in candidate generation. Base sampling is seeded only for
    reproducibility, not tuned for eval coverage.
    """
    tops = [item for item in wardrobe if item.category == "top"]
    bottoms = [item for item in wardrobe if item.category == "bottom"]
    layering = [item for item in wardrobe if item.category == "layering"]
    accessories = [item for item in wardrobe if item.category == "accessory"]

    owned_ids = {item.id for item in wardrobe}

    base: list[list[str]] = []
    for top, bottom in product(tops, bottoms):
        base.append([top.id, bottom.id])
        for layer in layering:
            base.append([top.id, bottom.id, layer.id])

    # Reserve room for accessory variants within the overall cap.
    max_base = max(max_candidates - max_accessory_variants, 1) if accessories else max_candidates
    if len(base) > max_base:
        base = random.Random(BASE_SAMPLE_SEED).sample(base, max_base)

    # Deterministic accessory variants: add each accessory to the first few base
    # outfits so the stylist sees with/without pairs to reason over.
    variants: list[list[str]] = []
    for outfit in base[: max_accessory_variants if accessories else 0]:
        for accessory in accessories:
            variants.append(outfit + [accessory.id])
            if len(variants) >= max_accessory_variants:
                break
        if len(variants) >= max_accessory_variants:
            break

    combined = (base + variants)[:max_candidates]

    candidates = [OutfitCandidate(items=items) for items in combined]
    for candidate in candidates:
        if not all(item_id in owned_ids for item_id in candidate.items):
            raise ValueError(f"Candidate contains item not in wardrobe: {candidate}")

    return candidates
