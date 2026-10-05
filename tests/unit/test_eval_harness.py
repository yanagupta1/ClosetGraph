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
    ]
    golden_path, wardrobe_path = _write_inputs(tmp_path, entries)

    rankings = {
        # top pick excludes b5 -> passes
        "office": [RankedOutfit(items=["t1", "b1"], rank=1, reasoning="x")],
        # top pick contains b1 -> fails the exclusion check
        "errands": [RankedOutfit(items=["t1", "b1"], rank=1, reasoning="x")],
        # judged entry -> collected, not hard-checked
        "bar": [RankedOutfit(items=["t1", "b5"], rank=1, reasoning="x")],
    }
    monkeypatch.setattr(eval_harness, "build_graph", lambda: FakeGraph(rankings))

    result = eval_harness.run_golden_set(str(golden_path), str(wardrobe_path))

    assert result["passed"] == ["e1"]
    assert [f["id"] for f in result["failed"]] == ["e2"]
    assert "b1" in result["failed"][0]["reason"]
    assert len(result["judged_pending"]) == 1
    assert result["judged_pending"][0]["id"] == "e3"
    assert result["judged_pending"][0]["judge_question"] == "Is the palette cohesive?"
    assert result["judged_pending"][0]["top_outfit"] == ["t1", "b5"]
