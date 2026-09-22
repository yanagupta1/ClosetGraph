from closetgraph.graph import build_graph
from closetgraph.models import RankedOutfit


def test_graph_generates_candidates_and_ranked_outfits(monkeypatch, sample_wardrobe):
    def fake_rank_outfits(candidates, wardrobe, occasion):
        assert len(candidates) == 12
        assert wardrobe == sample_wardrobe
        assert occasion == "team happy hour, casual bar"
        return [
            RankedOutfit(
                items=candidates[0].items,
                rank=1,
                reasoning="This is a coherent outfit for the occasion.",
            )
        ]

    monkeypatch.setattr("closetgraph.graph.rank_outfits", fake_rank_outfits)
    graph = build_graph()

    result = graph.invoke(
        {
            "wardrobe": sample_wardrobe,
            "occasion": "team happy hour, casual bar",
            "candidate_outfits": [],
            "ranked_outfits": [],
        }
    )

    assert len(result["candidate_outfits"]) == 12
    assert result["ranked_outfits"] == [
        RankedOutfit(
            items=["t1", "b1"],
            rank=1,
            reasoning="This is a coherent outfit for the occasion.",
        )
    ]


def test_graph_preserves_occasion_through_stylist_node(monkeypatch, sample_wardrobe):
    seen = {}

    def fake_rank_outfits(candidates, wardrobe, occasion):
        seen["occasion"] = occasion
        return [
            RankedOutfit(
                items=["t2", "b1", "l1"],
                rank=1,
                reasoning="This is polished enough for a client meeting.",
            )
        ]

    monkeypatch.setattr("closetgraph.graph.rank_outfits", fake_rank_outfits)
    graph = build_graph()

    result = graph.invoke(
        {
            "wardrobe": sample_wardrobe,
            "occasion": "first client meeting, conservative office",
            "candidate_outfits": [],
            "ranked_outfits": [],
        }
    )

    assert seen["occasion"] == "first client meeting, conservative office"
    assert result["ranked_outfits"][0].items == ["t2", "b1", "l1"]
