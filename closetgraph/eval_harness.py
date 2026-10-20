import json
import time

import yaml

from closetgraph.graph import build_graph
from closetgraph.models import OutfitCandidate, WardrobeItem
from closetgraph.retrieval import retrieve_style_guidance
from closetgraph.stylist import rank_outfits

# Heuristic keywords for the reasoning_flags_gap hard check (pure code, no LLM).
GAP_KEYWORDS = (
    "gap",
    "lack",
    "lacks",
    "lack of",
    "missing",
    "insufficient",
    "limitation",
    "does not have",
    "doesn't have",
    "do not have",
    "no suitable",
    "nothing suitable",
    "cannot satisfy",
    "wardrobe gap",
)
def _top_items(ranked):
    return set(ranked[0].items) if ranked else set()


def _best_rank_containing(ranked, item_id):
    ranks = [outfit.rank for outfit in ranked if item_id in outfit.items]
    return min(ranks) if ranks else None


def _check_hard_checks(hard_checks, ranked):
    """Return a failure reason string, or None if every hard check passes."""
    for key, value in hard_checks.items():
        if key == "top_rank_excludes":
            present = [i for i in value if i in _top_items(ranked)]
            if present:
                return f"top-ranked outfit should exclude {present}"
        elif key == "combination_appears_ranked":
            needed = set(value)
            if not any(needed.issubset(set(outfit.items)) for outfit in ranked):
                return f"no ranked outfit contains the combination {value}"
        elif key == "combination_not_top_rank":
            if set(value).issubset(_top_items(ranked)):
                return f"combination {value} should not be the top-ranked outfit"
        elif key == "prefers_over":
            higher = _best_rank_containing(ranked, value["higher"])
            lower = _best_rank_containing(ranked, value["lower"])
            if higher is None:
                return f"preferred item {value['higher']} does not appear in any ranked outfit"
            if lower is not None and not (higher < lower):
                return f"{value['higher']} should rank above {value['lower']} (ranks {higher} vs {lower})"
        elif key == "reasoning_flags_gap":
            text = " ".join(outfit.reasoning.lower() for outfit in ranked)
            if value and not any(keyword in text for keyword in GAP_KEYWORDS):
                return "reasoning did not flag a wardrobe gap"
        else:
            return f"unknown hard_check: {key}"
    return None


def run_golden_set(golden_set_path, wardrobe_path, delay_seconds: float = 0) -> dict:
    """
    Loads the golden set and wardrobe, runs the full graph for each
    entry's occasion, checks hard_checks in code (no LLM) for
    non-judged entries. For judged: true entries, skip the actual
    judge LLM call in this function, just collect the top-ranked
    outfit and judge_question into a separate list for manual/later
    review, do not implement LLM-judging yet, that's a follow-up once
    hard-check results are reviewed.
    Returns a dict: {passed: [...], failed: [...with reason...],
    judged_pending: [...]}.
    Print a summary: X/Y hard checks passed, Z judged cases pending.
    """
    with open(golden_set_path, encoding="utf-8") as handle:
        entries = yaml.safe_load(handle)
    with open(wardrobe_path, encoding="utf-8") as handle:
        wardrobe = [WardrobeItem(**item) for item in json.load(handle)]

    graph = build_graph()
    passed: list[str] = []
    failed: list[dict] = []
    judged_pending: list[dict] = []

    for index, entry in enumerate(entries):
        if delay_seconds and index:
            time.sleep(delay_seconds)
        result = graph.invoke(
            {
                "wardrobe": wardrobe,
                "occasion": entry["occasion"],
                "candidate_outfits": [],
                "ranked_outfits": [],
            }
        )
        ranked = result["ranked_outfits"]

        if entry.get("judged") is True:
            judged_pending.append(
                {
                    "id": entry["id"],
                    "occasion": entry["occasion"],
                    "judge_question": entry["judge_question"],
                    "top_outfit": ranked[0].items if ranked else None,
                }
            )
            continue

        reason = _check_hard_checks(entry["hard_checks"], ranked)
        if reason is None:
            passed.append(entry["id"])
        else:
            failed.append({"id": entry["id"], "reason": reason})

    hard_total = len(passed) + len(failed)
    print(
        f"{len(passed)}/{hard_total} hard checks passed, "
        f"{len(judged_pending)} judged cases pending"
    )
    return {"passed": passed, "failed": failed, "judged_pending": judged_pending}


def compare_pairwise(outfit_a, outfit_b, occasion, wardrobe, retrieved_guidance=None):
    """
    Compare the suitability of two specific outfits for one occasion by asking
    the stylist to rank ONLY those two. This measures A-vs-B suitability
    directly; it does not touch the main 20-candidate top 5.

    Returns {"winner": "A"|"B"|None, "ranked": [...]}. winner is whichever
    outfit the stylist ranks first; None if the ranking matches neither (e.g.
    the model returned unexpected ids).
    """
    candidates = [OutfitCandidate(items=outfit_a), OutfitCandidate(items=outfit_b)]
    if retrieved_guidance is None:
        retrieved_guidance = retrieve_style_guidance(occasion)
    ranked = rank_outfits(candidates, wardrobe, occasion, retrieved_guidance)

    winner = None
    if ranked:
        top = set(ranked[0].items)
        if top == set(outfit_a):
            winner = "A"
        elif top == set(outfit_b):
            winner = "B"
    return {
        "winner": winner,
        "ranked": [
            {"rank": o.rank, "items": o.items, "reasoning": o.reasoning}
            for o in ranked
        ],
    }


def run_pairwise_case(case, wardrobe) -> dict:
    """
    Run one pairwise compositional case. The 'favoring' occasion has a hard
    expectation (outfit B should be judged more suitable). The optional
    'reversal' occasion is recorded but NOT hard-asserted: there B simply
    should not automatically win, which is a softer, context-dependent signal
    better left to review than a pass/fail.
    """
    outfit_a = case["outfit_a"]
    outfit_b = case["outfit_b"]
    result = {"id": case["id"], "question": case.get("question")}

    favoring = compare_pairwise(outfit_a, outfit_b, case["occasion_favoring_b"], wardrobe)
    result["favoring"] = {
        "occasion": case["occasion_favoring_b"],
        "expected_winner": "B",
        "winner": favoring["winner"],
        "passed": favoring["winner"] == "B",
        "ranked": favoring["ranked"],
    }

    if case.get("occasion_reversal"):
        reversal = compare_pairwise(outfit_a, outfit_b, case["occasion_reversal"], wardrobe)
        result["reversal"] = {
            "occasion": case["occasion_reversal"],
            "note": "B should not automatically win here (recorded, not asserted)",
            "winner": reversal["winner"],
            "ranked": reversal["ranked"],
        }

    return result
