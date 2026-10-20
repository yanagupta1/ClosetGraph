from closetgraph.models import RankedOutfit
from closetgraph.validity import check_outfit_validity, validate_ranked_outfits


def _outfit(items, rank=1):
    return RankedOutfit(items=items, rank=rank, reasoning="n/a")


def test_valid_outfit_returns_no_violations(sample_wardrobe):
    outfit = _outfit(["t1", "b1", "l1"])

    assert check_outfit_validity(outfit, sample_wardrobe) == []


def test_unowned_id_returns_ownership_violation(sample_wardrobe):
    outfit = _outfit(["t1", "b1", "x9"])

    violations = check_outfit_validity(outfit, sample_wardrobe)

    assert any("unowned item id: x9" in v for v in violations)


def test_missing_bottom_returns_category_violation(sample_wardrobe):
    outfit = _outfit(["t1", "l1"])

    violations = check_outfit_validity(outfit, sample_wardrobe)

    assert any("expected exactly one bottom, found 0" in v for v in violations)


def test_two_tops_and_no_bottom_returns_category_violation(sample_wardrobe):
    outfit = _outfit(["t1", "t2"])

    violations = check_outfit_validity(outfit, sample_wardrobe)

    assert any("expected exactly one top, found 2" in v for v in violations)
    assert any("expected exactly one bottom, found 0" in v for v in violations)


def test_duplicate_item_id_returns_duplicate_violation(sample_wardrobe):
    outfit = _outfit(["t1", "b1", "b1"])

    violations = check_outfit_validity(outfit, sample_wardrobe)

    assert any("duplicate item id: b1" in v for v in violations)


def test_validate_ranked_outfits_keys_by_index_and_omits_clean(sample_wardrobe):
    outfits = [
        _outfit(["t1", "b1"], rank=1),       # valid
        _outfit(["t1", "b1", "x9"], rank=2),  # unowned id
        _outfit(["t2", "b2", "l1"], rank=3),  # valid
    ]

    result = validate_ranked_outfits(outfits, sample_wardrobe)

    assert set(result.keys()) == {1}
    assert any("unowned item id: x9" in v for v in result[1])
