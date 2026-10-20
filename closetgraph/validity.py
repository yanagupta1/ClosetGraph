from closetgraph.models import RankedOutfit, WardrobeItem


def check_outfit_validity(outfit: RankedOutfit, wardrobe: list[WardrobeItem]) -> list[str]:
    """
    Deterministic, no LLM call. Returns a list of violation strings;
    empty list means valid.
    Checks:
    1. Ownership: every item id in outfit.items must exist in wardrobe.
    2. Category sanity: exactly one "top" and exactly one "bottom"
       category item required. Layering items optional (zero or more).
    3. No duplicate item ids within one outfit.
    Does NOT evaluate styling quality, that's the Stylist's job, this
    is structural/ownership validity only.
    """
    violations: list[str] = []
    by_id = {item.id: item for item in wardrobe}

    # 1. Ownership
    for item_id in outfit.items:
        if item_id not in by_id:
            violations.append(f"unowned item id: {item_id}")

    # 3. No duplicate item ids
    seen: set[str] = set()
    for item_id in outfit.items:
        if item_id in seen:
            violations.append(f"duplicate item id: {item_id}")
        seen.add(item_id)

    # 2. Category sanity (counted over owned items only)
    top_count = sum(
        1 for item_id in outfit.items
        if item_id in by_id and by_id[item_id].category == "top"
    )
    bottom_count = sum(
        1 for item_id in outfit.items
        if item_id in by_id and by_id[item_id].category == "bottom"
    )
    if top_count != 1:
        violations.append(f"expected exactly one top, found {top_count}")
    if bottom_count != 1:
        violations.append(f"expected exactly one bottom, found {bottom_count}")

    return violations


def validate_ranked_outfits(ranked_outfits: list[RankedOutfit], wardrobe: list[WardrobeItem]) -> dict[int, list[str]]:
    """
    Runs check_outfit_validity on every ranked outfit. Returns
    {index: [violations]} only for outfits that have violations.
    Used standalone and later inside the eval harness.
    """
    result: dict[int, list[str]] = {}
    for index, outfit in enumerate(ranked_outfits):
        violations = check_outfit_validity(outfit, wardrobe)
        if violations:
            result[index] = violations
    return result
