import pytest
from pydantic import ValidationError

from closetgraph.models import OutfitCandidate, RankedOutfit, WardrobeItem


def test_wardrobe_item_accepts_supported_categories():
    item = WardrobeItem(id="t1", name="black tank top", category="top")

    assert item.id == "t1"
    assert item.name == "black tank top"
    assert item.category == "top"


def test_wardrobe_item_rejects_unknown_category():
    with pytest.raises(ValidationError):
        WardrobeItem(id="x1", name="red scarf", category="accessory")


def test_outfit_candidate_stores_item_ids():
    candidate = OutfitCandidate(items=["t1", "b1", "l1"])

    assert candidate.items == ["t1", "b1", "l1"]


def test_ranked_outfit_stores_rank_and_reasoning():
    outfit = RankedOutfit(
        items=["t2", "b1"],
        rank=1,
        reasoning="The silk blouse and trousers read polished.",
    )

    assert outfit.rank == 1
    assert "polished" in outfit.reasoning
