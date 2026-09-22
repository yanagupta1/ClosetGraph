import pytest

from closetgraph.models import WardrobeItem


@pytest.fixture
def sample_wardrobe():
    return [
        WardrobeItem(id="t1", name="black tank top", category="top"),
        WardrobeItem(id="t2", name="white silk blouse", category="top"),
        WardrobeItem(id="b1", name="black tailored trousers", category="bottom"),
        WardrobeItem(id="b2", name="dark wash jeans", category="bottom"),
        WardrobeItem(id="l1", name="black structured blazer", category="layering"),
        WardrobeItem(id="l2", name="beige knit cardigan", category="layering"),
    ]
