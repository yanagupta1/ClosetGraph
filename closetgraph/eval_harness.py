import json
import time

import yaml

from closetgraph.graph import build_graph
from closetgraph.models import OutfitCandidate, WardrobeItem
from closetgraph.retrieval import retrieve_style_guidance
from closetgraph.stylist import rank_outfits
from closetgraph.validity import validate_ranked_outfits

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
    Loads the golden set and wardrobe, runs the full graph for each entry's
    occasion, and evaluates by entry `type`:

    - ranking_preference (and any untyped entry with hard_checks): the
      hard_checks are asserted in code (no LLM) -> passed / failed.
    - behavioral (and legacy judged: true): collected for rubric/manual review,
      NOT pass/fail. The automated rubric judge is a later follow-up.

    Hard constraints (ownership, one top + one bottom, no duplicates) are always
    enforced on every ranked output via validity.py and reported separately.

    Returns {passed, failed, behavioral_pending, constraint_violations}. Prints
    a summary: X/Y ranking checks passed, N behavioral collected, C with
    structural violations.
    """
    with open(golden_set_path, encoding="utf-8") as handle:
        entries = yaml.safe_load(handle)
    with open(wardrobe_path, encoding="utf-8") as handle:
        wardrobe = [WardrobeItem(**item) for item in json.load(handle)]

    graph = build_graph()
    passed: list[str] = []
    failed: list[dict] = []
    behavioral_pending: list[dict] = []
    constraint_violations: list[dict] = []

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

        # Hard constraints: structural validity on every output, regardless of type.
        violations = validate_ranked_outfits(ranked, wardrobe)
        if violations:
            constraint_violations.append({"id": entry["id"], "violations": violations})

        if entry.get("type") == "behavioral" or entry.get("judged") is True:
            behavioral_pending.append(
                {
                    "id": entry["id"],
                    "occasion": entry["occasion"],
                    "notes": entry.get("notes"),
                    "judge_question": entry.get("judge_question"),
                    "expectation": entry.get("hard_checks"),
                    "top_outfit": ranked[0].items if ranked else None,
                    "top_reasoning": ranked[0].reasoning if ranked else None,
                }
            )
            continue

        reason = _check_hard_checks(entry["hard_checks"], ranked)
        if reason is None:
            passed.append(entry["id"])
        else:
            failed.append({"id": entry["id"], "reason": reason})

    ranking_total = len(passed) + len(failed)
    print(
        f"{len(passed)}/{ranking_total} ranking checks passed, "
        f"{len(behavioral_pending)} behavioral collected, "
        f"{len(constraint_violations)} with structural violations"
    )
    return {
        "passed": passed,
        "failed": failed,
        "behavioral_pending": behavioral_pending,
        "constraint_violations": constraint_violations,
    }


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


def run_pairwise_set(
    pairwise_set_path, wardrobe_path, repeats: int = 1, delay_seconds: float = 0
) -> list[dict]:
    """
    Budget-aware diagnostic runner. Runs each pairwise case `repeats` times and
    records every individual run, so nondeterminism is visible and two models
    can be compared on identical cases. Keep the set small and repeats low to
    respect the free-tier token budget; throttle with delay_seconds.

    Returns a list of {id, question, runs: [<run_pairwise_case result>, ...],
    favoring_wins: int} - favoring_wins counts how many of the repeats put B
    first for the favoring occasion.
    """
    with open(pairwise_set_path, encoding="utf-8") as handle:
        cases = yaml.safe_load(handle)
    with open(wardrobe_path, encoding="utf-8") as handle:
        wardrobe = [WardrobeItem(**item) for item in json.load(handle)]

    results: list[dict] = []
    first_call = True
    for case in cases:
        runs: list[dict] = []
        for _ in range(repeats):
            if delay_seconds and not first_call:
                time.sleep(delay_seconds)
            first_call = False
            runs.append(run_pairwise_case(case, wardrobe))
        favoring_wins = sum(1 for r in runs if r["favoring"]["passed"])
        results.append(
            {
                "id": case["id"],
                "question": case.get("question"),
                "favoring_wins": favoring_wins,
                "repeats": repeats,
                "runs": runs,
            }
        )
        print(f"{case['id']}: B favored {favoring_wins}/{repeats} runs")

    return results
