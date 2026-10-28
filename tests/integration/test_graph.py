from closetgraph.graph import build_graph, weather_lookup_node
from closetgraph.models import RankedOutfit, WeatherInput


def test_weather_lookup_node_resolves_user_weather_without_api(monkeypatch):
    # User-provided weather resolves without any fetch call.
    monkeypatch.setattr(
        "closetgraph.graph.fetch_open_meteo",
        lambda location: (_ for _ in ()).throw(AssertionError("no API expected")),
    )
    state = {"weather_input": WeatherInput(temperature_c=5, condition="rainy")}

    out = weather_lookup_node(state)

    assert out["weather_context"].source == "user"
    assert out["weather_context"].condition == "rainy"
    assert out["weather_error"] is None


def test_weather_lookup_node_no_input_is_noop(monkeypatch):
    monkeypatch.setattr(
        "closetgraph.graph.fetch_open_meteo",
        lambda location: (_ for _ in ()).throw(AssertionError("no API expected")),
    )

    out = weather_lookup_node({"weather_input": None})

    assert out["weather_context"] is None
    assert out["weather_error"] is None


def test_weather_lookup_node_surfaces_fetch_error(monkeypatch):
    def boom(location):
        raise OSError("network down")

    monkeypatch.setattr("closetgraph.graph.fetch_open_meteo", boom)

    out = weather_lookup_node({"weather_input": WeatherInput(location="Paris")})

    assert out["weather_context"] is None  # no fabrication
    assert "network down" in out["weather_error"]


def test_graph_generates_candidates_and_ranked_outfits(monkeypatch, sample_wardrobe):
    def fake_rank_outfits(candidates, wardrobe, occasion, retrieved_guidance=None):
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
    monkeypatch.setattr("closetgraph.graph.retrieve_style_guidance", lambda occasion: [])
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

    def fake_rank_outfits(candidates, wardrobe, occasion, retrieved_guidance=None):
        seen["occasion"] = occasion
        return [
            RankedOutfit(
                items=["t2", "b1", "l1"],
                rank=1,
                reasoning="This is polished enough for a client meeting.",
            )
        ]

    monkeypatch.setattr("closetgraph.graph.rank_outfits", fake_rank_outfits)
    monkeypatch.setattr("closetgraph.graph.retrieve_style_guidance", lambda occasion: [])
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
