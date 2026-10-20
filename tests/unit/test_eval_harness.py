import json

import yaml

import closetgraph.eval_harness as eval_harness
from closetgraph.models import RankedOutfit


class FakeGraph:
    def __init__(self, rankings_by_occasion):
        self.rankings_by_occasion = rankings_by_occasion

    def invoke(self, state):
        state["ranked_outfits"] = self.rankings_by_occasion[state["occasion"]]
        return state


def _write_inputs(tmp_path, entries):
    golden_path = tmp_path / "golden.yaml"
    golden_path.write_text(yaml.safe_dump(entries), encoding="utf-8")
    wardrobe_path = tmp_path / "wardrobe.json"
    wardrobe_path.write_text(
        json.dumps(
            [
                {"id": "t1", "name": "tank", "category": "top"},
                {"id": "b1", "name": "trousers", "category": "bottom"},
                {"id": "b5", "name": "relaxed denim", "category": "bottom"},
            ]
        ),
        encoding="utf-8",
    )
    return golden_path, wardrobe_path


def test_run_golden_set_passes_fails_and_collects_judged(tmp_path, monkeypatch):
    entries = [
        {
            "id": "e1",
            "category": "occasion_mapping",
            "occasion": "office",
            "hard_checks": {"top_rank_excludes": ["b5"]},
        },
        {
            "id": "e2",
            "category": "occasion_mapping",
            "occasion": "errands",
            "hard_checks": {"top_rank_excludes": ["b1"]},
        },
        {
            "id": "e3",
            "category": "color_cohesion",
            "judged": True,
            "occasion": "bar",
            "judge_question": "Is the palette cohesive?",
        },
        {
            # behavioral type -> collected, not hard-checked, even with hard_checks
            "id": "e4",
            "category": "layering_elevation",
            "type": "behavioral",
            "occasion": "gala",
            "hard_checks": {"combination_appears_ranked": ["t1", "b5"]},
        },
    ]
    golden_path, wardrobe_path = _write_inputs(tmp_path, entries)

    rankings = {
        # top pick excludes b5 -> passes
        "office": [RankedOutfit(items=["t1", "b1"], rank=1, reasoning="x")],
        # top pick contains b1 -> fails the exclusion check
        "errands": [RankedOutfit(items=["t1", "b1"], rank=1, reasoning="x")],
        # judged entry -> collected, not hard-checked
        "bar": [RankedOutfit(items=["t1", "b5"], rank=1, reasoning="x")],
        # behavioral entry -> collected regardless of the combo appearing
        "gala": [RankedOutfit(items=["t1", "b1"], rank=1, reasoning="x")],
    }
    monkeypatch.setattr(eval_harness, "build_graph", lambda: FakeGraph(rankings))

    result = eval_harness.run_golden_set(str(golden_path), str(wardrobe_path))

    assert result["passed"] == ["e1"]
    assert [f["id"] for f in result["failed"]] == ["e2"]
    assert "b1" in result["failed"][0]["reason"]
    # e3 (judged) and e4 (behavioral) both collected, not pass/failed
    collected = {b["id"] for b in result["behavioral_pending"]}
    assert collected == {"e3", "e4"}
    e3 = next(b for b in result["behavioral_pending"] if b["id"] == "e3")
    assert e3["judge_question"] == "Is the palette cohesive?"
    assert e3["top_outfit"] == ["t1", "b5"]
    # all outputs structurally valid (one top + one bottom) -> no violations
    assert result["constraint_violations"] == []


def test_run_pairwise_case_asserts_favoring_and_records_reversal(monkeypatch, sample_wardrobe):
    # Stub the stylist: B (with blazer) wins the client meeting, A wins the picnic.
    def fake_rank_outfits(candidates, wardrobe, occasion, retrieved_guidance=None):
        outfit_b = ["t1", "b1", "l1"]
        if "client meeting" in occasion:
            order = [c for c in candidates if c.items == outfit_b] + [
                c for c in candidates if c.items != outfit_b
            ]
        else:  # casual context -> the simpler outfit A wins
            order = [c for c in candidates if c.items != outfit_b] + [
                c for c in candidates if c.items == outfit_b
            ]
        return [
            RankedOutfit(items=c.items, rank=i + 1, reasoning="stub")
            for i, c in enumerate(order)
        ]

    monkeypatch.setattr(eval_harness, "rank_outfits", fake_rank_outfits)
    monkeypatch.setattr(eval_harness, "retrieve_style_guidance", lambda occasion: [])

    case = {
        "id": "p01",
        "question": "Does a blazer improve professional suitability?",
        "outfit_a": ["t1", "b1"],
        "outfit_b": ["t1", "b1", "l1"],
        "occasion_favoring_b": "conservative client meeting",
        "occasion_reversal": "casual outdoor summer gathering",
    }

    result = eval_harness.run_pairwise_case(case, sample_wardrobe)

    assert result["favoring"]["winner"] == "B"
    assert result["favoring"]["passed"] is True
    # Reversal is recorded, not asserted: here A wins, so B did not auto-win.
    assert result["reversal"]["winner"] == "A"


def test_run_pairwise_set_repeats_and_counts_favoring_wins(tmp_path, monkeypatch):
    # B always wins the favoring occasion -> favoring_wins should equal repeats.
    def fake_rank_outfits(candidates, wardrobe, occasion, retrieved_guidance=None):
        outfit_b = ["t1", "b1", "l1"]
        order = [c for c in candidates if c.items == outfit_b] + [
            c for c in candidates if c.items != outfit_b
        ]
        return [
            RankedOutfit(items=c.items, rank=i + 1, reasoning="stub")
            for i, c in enumerate(order)
        ]

    monkeypatch.setattr(eval_harness, "rank_outfits", fake_rank_outfits)
    monkeypatch.setattr(eval_harness, "retrieve_style_guidance", lambda occasion: [])

    pairwise_path = tmp_path / "pairwise.yaml"
    pairwise_path.write_text(
        yaml.safe_dump(
            [
                {
                    "id": "p01",
                    "outfit_a": ["t1", "b1"],
                    "outfit_b": ["t1", "b1", "l1"],
                    "occasion_favoring_b": "client meeting",
                }
            ]
        ),
        encoding="utf-8",
    )
    wardrobe_path = tmp_path / "wardrobe.json"
    wardrobe_path.write_text(
        json.dumps(
            [
                {"id": "t1", "name": "tank", "category": "top"},
                {"id": "b1", "name": "trousers", "category": "bottom"},
                {"id": "l1", "name": "blazer", "category": "layering"},
            ]
        ),
        encoding="utf-8",
    )

    results = eval_harness.run_pairwise_set(
        str(pairwise_path), str(wardrobe_path), repeats=3
    )

    assert len(results) == 1
    assert results[0]["id"] == "p01"
    assert results[0]["repeats"] == 3
    assert results[0]["favoring_wins"] == 3
    assert len(results[0]["runs"]) == 3
