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


def test_generate_candidates_adds_deterministic_accessory_variants():
    wardrobe = [
        WardrobeItem(id="t1", name="tank", category="top"),
        WardrobeItem(id="b1", name="trousers", category="bottom"),
        WardrobeItem(id="l1", name="blazer", category="layering"),
        WardrobeItem(id="a1", name="scarf", category="accessory"),
    ]

    candidates = [c.items for c in generate_candidates(wardrobe, max_accessory_variants=2)]

    # The accessory is available as a variant of a base outfit, not gated.
    with_scarf = [c for c in candidates if "a1" in c]
    assert with_scarf, "expected at least one accessory variant"
    for variant in with_scarf:
        assert variant[-1] == "a1"
        assert variant[:-1] in candidates  # it is a base outfit plus the accessory


def test_generate_candidates_pool_stays_bounded_as_wardrobe_grows(sample_wardrobe):
    # Add enough items that the raw Cartesian product far exceeds the cap.
    big = list(sample_wardrobe) + [
        WardrobeItem(id=f"t{i}", name=f"top {i}", category="top") for i in range(5, 15)
    ] + [WardrobeItem(id="a1", name="scarf", category="accessory")]

    candidates = generate_candidates(big, max_candidates=20, max_accessory_variants=4)

    assert len(candidates) <= 20
    assert any("a1" in c.items for c in candidates)
