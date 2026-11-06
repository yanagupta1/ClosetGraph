import json

import yaml

import closetgraph.eval_harness as eval_harness
from closetgraph.models import OutfitCandidate, RankedOutfit


class FakeGraph:
    def __init__(self, rankings_by_occasion, candidates_by_occasion=None):
        self.rankings_by_occasion = rankings_by_occasion
        self.candidates_by_occasion = candidates_by_occasion or {}
        self.seen_states = []

    def invoke(self, state):
        self.seen_states.append(dict(state))
        occasion = state["occasion"]
        ranked = self.rankings_by_occasion[occasion]
        state["ranked_outfits"] = ranked
        # Default the candidate pool to the ranked outfits so checks are
        # applicable unless a test specifies a narrower pool.
        state["candidate_outfits"] = self.candidates_by_occasion.get(
            occasion, [OutfitCandidate(items=o.items) for o in ranked]
        )
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
    # coverage: both ranking checks applicable, pass rate over applicable only
    cov = result["coverage"]
    assert cov["applicable"] == 2
    assert cov["inapplicable"] == 0
    assert cov["pass_rate_applicable"] == 0.5
    assert cov["applicable_coverage"] == 1.0


def test_run_golden_set_injects_fixture_weather_without_api(tmp_path, monkeypatch):
    entries = [
        {   # carries fixed weather -> must reach the graph state, no API call
            "id": "w1",
            "category": "occasion_mapping",
            "type": "ranking_preference",
            "occasion": "patio dinner",
            "hard_checks": {"top_rank_excludes": ["b5"]},
            "weather": {"temperature_c": 6, "condition": "rainy"},
        },
        {   # no weather block -> weather_input stays None (Stage 2 behaviour)
            "id": "w2",
            "category": "occasion_mapping",
            "type": "ranking_preference",
            "occasion": "office",
            "hard_checks": {"top_rank_excludes": ["b5"]},
        },
    ]
    golden_path, wardrobe_path = _write_inputs(tmp_path, entries)
    rankings = {
        "patio dinner": [RankedOutfit(items=["t1", "b1"], rank=1, reasoning="x")],
        "office": [RankedOutfit(items=["t1", "b1"], rank=1, reasoning="x")],
    }
    fake = FakeGraph(rankings)
    monkeypatch.setattr(eval_harness, "build_graph", lambda: fake)

    eval_harness.run_golden_set(str(golden_path), str(wardrobe_path))

    by_occasion = {s["occasion"]: s["weather_input"] for s in fake.seen_states}
    assert by_occasion["patio dinner"].temperature_c == 6
    assert by_occasion["patio dinner"].condition == "rainy"
    assert by_occasion["office"] is None


def test_run_golden_set_isolates_per_case_errors(tmp_path, monkeypatch):
    # One case raises (e.g. truncated JSON); the other must still be evaluated.
    entries = [
        {"id": "boom", "category": "occasion_mapping", "type": "ranking_preference",
         "occasion": "explode", "hard_checks": {"top_rank_excludes": ["b5"]}},
        {"id": "ok", "category": "occasion_mapping", "type": "ranking_preference",
         "occasion": "office", "hard_checks": {"top_rank_excludes": ["b5"]}},
    ]
    golden_path, wardrobe_path = _write_inputs(tmp_path, entries)

    class ExplodingGraph:
        def invoke(self, state):
            if state["occasion"] == "explode":
                raise ValueError("Unterminated string")
            state["ranked_outfits"] = [RankedOutfit(items=["t1", "b1"], rank=1, reasoning="x")]
            state["candidate_outfits"] = [OutfitCandidate(items=["t1", "b1"])]
            return state

    monkeypatch.setattr(eval_harness, "build_graph", lambda: ExplodingGraph())

    result = eval_harness.run_golden_set(str(golden_path), str(wardrobe_path))

    assert [e["id"] for e in result["errored"]] == ["boom"]
    assert result["passed"] == ["ok"]  # the good case still evaluated
    assert "Unterminated string" in result["errored"][0]["error"]


def test_run_golden_set_marks_inapplicable_missing_item_vs_combination(tmp_path, monkeypatch):
    entries = [
        {   # references b5, which is NOT in the candidate pool -> missing_item
            "id": "mi",
            "category": "garment_denim",
            "type": "ranking_preference",
            "occasion": "brunch",
            "hard_checks": {"prefers_over": {"higher": "b1", "lower": "b5"}},
        },
        {   # t1 and b1 are both in the pool, but never together in one outfit
            "id": "mc",
            "category": "layering_elevation",
            "type": "ranking_preference",
            "occasion": "office",
            "hard_checks": {"combination_appears_ranked": ["t1", "b1"]},
        },
    ]
    golden_path, wardrobe_path = _write_inputs(tmp_path, entries)

    rankings = {
        "brunch": [RankedOutfit(items=["t1", "b1"], rank=1, reasoning="x")],
        "office": [RankedOutfit(items=["t1", "b1"], rank=1, reasoning="x")],
    }
    candidates = {
        # pool has b1 but no b5 at all
        "brunch": [OutfitCandidate(items=["t1", "b1"])],
        # pool has t1 and b1 as items, but never in the same outfit
        "office": [OutfitCandidate(items=["t1"]), OutfitCandidate(items=["b1"])],
    }
    monkeypatch.setattr(
        eval_harness, "build_graph", lambda: FakeGraph(rankings, candidates)
    )

    result = eval_harness.run_golden_set(str(golden_path), str(wardrobe_path))

    assert result["passed"] == []
    assert result["failed"] == []
    kinds = {i["id"]: i["kind"] for i in result["inapplicable"]}
    assert kinds == {"mi": "missing_item", "mc": "missing_combination"}
    # nothing applicable -> coverage 0, pass rate undefined
    cov = result["coverage"]
    assert cov["applicable"] == 0
    assert cov["inapplicable"] == 2
    assert cov["applicable_coverage"] == 0.0
    assert cov["pass_rate_applicable"] is None


def test_run_pairwise_case_asserts_favoring_and_records_reversal(monkeypatch, sample_wardrobe):
    # Stub the stylist: B (with blazer) wins the client meeting, A wins the picnic.
    def fake_rank_outfits(candidates, wardrobe, occasion, retrieved_guidance=None, weather_context=None):
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
    monkeypatch.setattr(eval_harness, "retrieve_style_guidance", lambda occasion, weather_context=None: [])

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
    def fake_rank_outfits(candidates, wardrobe, occasion, retrieved_guidance=None, weather_context=None):
        outfit_b = ["t1", "b1", "l1"]
        order = [c for c in candidates if c.items == outfit_b] + [
            c for c in candidates if c.items != outfit_b
        ]
        return [
            RankedOutfit(items=c.items, rank=i + 1, reasoning="stub")
            for i, c in enumerate(order)
        ]

    monkeypatch.setattr(eval_harness, "rank_outfits", fake_rank_outfits)
    monkeypatch.setattr(eval_harness, "retrieve_style_guidance", lambda occasion, weather_context=None: [])

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


def test_run_golden_set_judges_rubric_color_case(tmp_path, monkeypatch):
    from closetgraph.judge import JudgeVerdict

    entries = [
        {
            "id": "c1",
            "category": "color_cohesion",
            "type": "behavioral",
            "judged": True,
            "occasion": "bar",
            "judge_question": "Cohesive palette?",
            "rubric": "Pass if colours are intentional.",
        },
    ]
    golden_path, wardrobe_path = _write_inputs(tmp_path, entries)

    rankings = {"bar": [RankedOutfit(items=["t1", "b1"], rank=1, reasoning="neutral base")]}
    monkeypatch.setattr(eval_harness, "build_graph", lambda: FakeGraph(rankings))

    captured = {}

    def fake_judge(outfit, reasoning, occasion, rubric):
        captured["outfit"] = outfit
        captured["rubric"] = rubric
        return JudgeVerdict(passed=True, score=5, rationale="cohesive neutrals")

    monkeypatch.setattr(eval_harness, "judge_outfit", fake_judge)

    result = eval_harness.run_golden_set(str(golden_path), str(wardrobe_path))

    assert len(result["behavioral_pending"]) == 1
    record = result["behavioral_pending"][0]
    assert record["verdict"] == {"passed": True, "score": 5, "rationale": "cohesive neutrals"}
    # the judge received resolved garment names, not raw ids
    assert "tank" in captured["outfit"]
    assert captured["rubric"] == "Pass if colours are intentional."


# --- weather pairwise (mocked stylist, no Groq) ---

_WX_WARDROBE = [
    {"id": "t4", "name": "navy fitted long-sleeve top", "category": "top"},
    {"id": "b1", "name": "black tailored trousers", "category": "bottom"},
    {"id": "l3", "name": "camel wool trench coat", "category": "layering"},
    {"id": "l1", "name": "black structured blazer", "category": "layering"},
]


def _wx_wardrobe_items():
    from closetgraph.models import WardrobeItem

    return [WardrobeItem(**i) for i in _WX_WARDROBE]


def test_weather_pairwise_reports_ranking_and_consistency_separately(monkeypatch):
    # Stylist: the trench (l3) wins when cold, the base outfit wins when warm.
    def fake_rank_outfits(candidates, wardrobe, occasion, retrieved_guidance=None, weather_context=None):
        cold = weather_context is not None and (weather_context.temperature_c or 99) < 10
        with_trench = next(c for c in candidates if "l3" in c.items)
        without = next(c for c in candidates if "l3" not in c.items)
        order = [with_trench, without] if cold else [without, with_trench]
        return [
            RankedOutfit(items=c.items, rank=i + 1, reasoning="fine")
            for i, c in enumerate(order)
        ]

    monkeypatch.setattr(eval_harness, "rank_outfits", fake_rank_outfits)
    monkeypatch.setattr(eval_harness, "retrieve_style_guidance", lambda occasion, weather_context=None: [])

    case = {
        "id": "wp01",
        "outfit_a": ["t4", "b1"],
        "outfit_b": ["t4", "b1", "l3"],
        "occasion": "weekend walk",
        "conditions": [
            {"label": "cold", "weather": {"temperature_c": 2}, "expected_winner": "B"},
            {"label": "warm", "weather": {"temperature_c": 26}, "expected_winner": "A"},
        ],
    }

    result = eval_harness.run_weather_pairwise_case(case, _wx_wardrobe_items())

    assert result["ranking_passed"] is True  # winner flipped with weather, as expected
    assert result["garment_consistent"] is True
    assert [c["winner"] for c in result["conditions"]] == ["B", "A"]


def test_weather_pairwise_flags_garment_inconsistency_without_failing_ranking(monkeypatch):
    # Winner is correct, but the reasoning name-drops a blazer not in the outfit.
    def fake_rank_outfits(candidates, wardrobe, occasion, retrieved_guidance=None, weather_context=None):
        with_trench = next(c for c in candidates if "l3" in c.items)
        without = next(c for c in candidates if "l3" not in c.items)
        return [
            RankedOutfit(items=with_trench.items, rank=1, reasoning="the blazer adds polish"),
            RankedOutfit(items=without.items, rank=2, reasoning="fine"),
        ]

    monkeypatch.setattr(eval_harness, "rank_outfits", fake_rank_outfits)
    monkeypatch.setattr(eval_harness, "retrieve_style_guidance", lambda occasion, weather_context=None: [])

    case = {
        "id": "wp-bad",
        "outfit_a": ["t4", "b1"],
        "outfit_b": ["t4", "b1", "l3"],
        "occasion": "walk",
        "conditions": [
            {"label": "cold", "weather": {"temperature_c": 2}, "expected_winner": "B"},
        ],
    }

    result = eval_harness.run_weather_pairwise_case(case, _wx_wardrobe_items())

    assert result["ranking_passed"] is True  # ranking is right
    assert result["garment_consistent"] is False  # but reasoning is not - reported apart
    assert "blazer" in result["conditions"][0]["reasoning_offenders"]


def test_run_routing_set_scores_against_labels(tmp_path, monkeypatch):
    cases = [
        {"id": "r1", "occasion": "office", "expected_research": False, "reason": "known"},
        {"id": "r2", "occasion": "hunt ball", "expected_research": True, "reason": "uncommon"},
        {"id": "r3", "occasion": "gala", "expected_research": True, "reason": "formal"},
    ]
    path = tmp_path / "routing.yaml"
    path.write_text(yaml.safe_dump(cases), encoding="utf-8")

    # Stub supervise: correct on r1/r2, wrong on r3 (says skip) -> 2/3.
    def fake_supervise(occasion):
        needed = occasion in ("hunt ball",)
        return {"research_needed": needed, "research_query": None, "research_reason": "x"}

    monkeypatch.setattr(eval_harness, "supervise", fake_supervise)

    result = eval_harness.run_routing_set(str(path))

    assert result["total"] == 3
    assert result["passed"] == 2
    assert result["accuracy"] == 2 / 3
    assert result["reason_quality"] is None  # not judged unless requested
    wrong = [c for c in result["cases"] if not c["passed"]]
    assert [c["id"] for c in wrong] == ["r3"]


def test_run_routing_set_judges_reason_quality_when_requested(tmp_path, monkeypatch):
    from closetgraph.judge import JudgeVerdict

    cases = [
        {"id": "r1", "occasion": "office", "expected_research": False, "reason": "known"},
        {"id": "r2", "occasion": "hunt ball", "expected_research": True, "reason": "uncommon"},
    ]
    path = tmp_path / "routing.yaml"
    path.write_text(yaml.safe_dump(cases), encoding="utf-8")

    monkeypatch.setattr(
        eval_harness,
        "supervise",
        lambda occasion: {
            "research_needed": occasion == "hunt ball",
            "research_query": None,
            "research_reason": "a specific reason",
        },
    )
    monkeypatch.setattr(
        eval_harness,
        "judge_routing_reason",
        lambda occasion, decision, reason: JudgeVerdict(
            passed=True, score=4, rationale="specific and correct"
        ),
    )

    result = eval_harness.run_routing_set(str(path), judge_reasons=True)

    assert result["reason_quality"] == {"passed": 2, "judged": 2}
    assert all("reason_verdict" in c for c in result["cases"])


def test_check_reasoning_garments_detects_absent_garment():
    wardrobe = _wx_wardrobe_items()
    # outfit has the trench but reasoning mentions a blazer (present in wardrobe,
    # not in this outfit) -> flagged.
    offenders = eval_harness.check_reasoning_garments(
        ["t4", "b1", "l3"], "the blazer and trench look sharp", wardrobe
    )
    assert "blazer" in offenders
    assert "trench" not in offenders  # trench IS in the outfit
