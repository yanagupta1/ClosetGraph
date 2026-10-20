from closetgraph.candidates import generate_candidates
from closetgraph.models import WardrobeItem


def test_generate_candidates_creates_top_bottom_pairs_without_layers():
    wardrobe = [
        WardrobeItem(id="t1", name="black tank top", category="top"),
        WardrobeItem(id="t2", name="white blouse", category="top"),
        WardrobeItem(id="b1", name="black trousers", category="bottom"),
    ]

    candidates = generate_candidates(wardrobe)

    assert [candidate.items for candidate in candidates] == [
        ["t1", "b1"],
        ["t2", "b1"],
    ]


def test_generate_candidates_adds_layering_variants(sample_wardrobe):
    candidates = generate_candidates(sample_wardrobe)

    candidate_items = [candidate.items for candidate in candidates]

    assert ["t1", "b1"] in candidate_items
    assert ["t1", "b1", "l1"] in candidate_items
    assert ["t1", "b1", "l2"] in candidate_items
    assert len(candidates) == 12


def test_generate_candidates_validates_all_ids_are_owned(sample_wardrobe):
    owned_ids = {item.id for item in sample_wardrobe}

    candidates = generate_candidates(sample_wardrobe)

    assert candidates
    for candidate in candidates:
        assert set(candidate.items).issubset(owned_ids)


def test_generate_candidates_samples_deterministically_when_over_cap(sample_wardrobe):
    first = generate_candidates(sample_wardrobe, max_candidates=5)
    second = generate_candidates(sample_wardrobe, max_candidates=5)

    assert len(first) == 5
    # Seeded sampling must be reproducible across calls so eval runs are stable.
    assert [c.items for c in first] == [c.items for c in second]

    owned_ids = {item.id for item in sample_wardrobe}
    for candidate in first:
        assert set(candidate.items).issubset(owned_ids)
