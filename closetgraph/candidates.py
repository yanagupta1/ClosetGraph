import random
from itertools import product

from closetgraph.models import OutfitCandidate, WardrobeItem


def generate_candidates(
    wardrobe: list[WardrobeItem], max_candidates: int = 20
) -> list[OutfitCandidate]:
    tops = [item for item in wardrobe if item.category == "top"]
    bottoms = [item for item in wardrobe if item.category == "bottom"]
    layering = [item for item in wardrobe if item.category == "layering"]

    owned_ids = {item.id for item in wardrobe}
    candidates: list[OutfitCandidate] = []

    for top, bottom in product(tops, bottoms):
        candidates.append(OutfitCandidate(items=[top.id, bottom.id]))

        for layer in layering:
            candidates.append(OutfitCandidate(items=[top.id, bottom.id, layer.id]))

    for candidate in candidates:
        if not all(item_id in owned_ids for item_id in candidate.items):
            raise ValueError(f"Candidate contains item not in wardrobe: {candidate}")

    if len(candidates) > max_candidates:
        candidates = random.sample(candidates, max_candidates)

    return candidates
