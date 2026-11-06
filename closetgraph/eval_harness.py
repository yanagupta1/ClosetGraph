import json
import time

import yaml

from closetgraph.graph import build_graph
from closetgraph.judge import judge_outfit, judge_routing_reason
from closetgraph.models import (
    OutfitCandidate,
    WardrobeItem,
    WeatherContext,
    WeatherInput,
)
from closetgraph.retrieval import retrieve_style_guidance
from closetgraph.stylist import rank_outfits
from closetgraph.supervisor import supervise
from closetgraph.validity import validate_ranked_outfits


def run_routing_set(
    routing_set_path, delay_seconds: float = 0, judge_reasons: bool = False
) -> dict:
    """
    Evaluate the supervisor in isolation: run supervise(occasion) only, with no
    research agent and no stylist, and compare its routing to the labels.

    Reports two things separately: decision accuracy (routing matches the label)
    and, when judge_reasons is set, reason quality (an LLM judges whether the
    supervisor's stated reason is a specific, correct justification, not just the
    yes/no). Deterministic skips are reproducible; classifier and judge cases
    make LLM calls (throttle with delay_seconds).
    """
    with open(routing_set_path, encoding="utf-8") as handle:
        cases = yaml.safe_load(handle)

    results = []
    first = True
    for case in cases:
        if delay_seconds and not first:
            time.sleep(delay_seconds)
        first = False
        decision = supervise(case["occasion"])
        passed = decision["research_needed"] == case["expected_research"]
        record = {
            "id": case["id"],
            "occasion": case["occasion"],
            "expected_research": case["expected_research"],
            "actual_research": decision["research_needed"],
            "passed": passed,
            "label_reason": case.get("reason"),
            "supervisor_reason": decision.get("research_reason"),
            "research_query": decision.get("research_query"),
        }
        if judge_reasons:
            if delay_seconds:
                time.sleep(delay_seconds)
            try:
                verdict = judge_routing_reason(
                    case["occasion"],
                    "research" if decision["research_needed"] else "skip",
                    decision.get("research_reason"),
                )
                record["reason_verdict"] = verdict.model_dump()
            except Exception as error:
                record["reason_judge_error"] = str(error)
        results.append(record)
        if not passed:
            print(
                f"  MISMATCH {case['id']}: expected {case['expected_research']}, "
                f"got {decision['research_needed']} :: {case['occasion']}"
            )

    passed = sum(1 for r in results if r["passed"])
    total = len(results)
    accuracy = passed / total if total else None
    # False positive: predicted research when the label says skip. False
    # negative: predicted skip when the label says research.
    false_positives = [
        r["id"] for r in results if r["actual_research"] and not r["expected_research"]
    ]
    false_negatives = [
        r["id"] for r in results if not r["actual_research"] and r["expected_research"]
    ]
    acc = f"{accuracy:.0%}" if total else "n/a"
    print(f"routing decisions: {passed}/{total} correct ({acc})")
    print(
        f"  false positives (researched a skip): {len(false_positives)} {false_positives}"
    )
    print(
        f"  false negatives (skipped a research): {len(false_negatives)} {false_negatives}"
    )

    reason_quality = None
    judged = [r for r in results if "reason_verdict" in r]
    if judged:
        good = sum(1 for r in judged if r["reason_verdict"]["passed"])
        reason_quality = {"passed": good, "judged": len(judged)}
        print(f"routing reasons: {good}/{len(judged)} judged adequate")

    return {
        "accuracy": accuracy,
        "passed": passed,
        "total": total,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "reason_quality": reason_quality,
        "cases": results,
    }


def _weather_input_from_entry(entry):
    """Build a WeatherInput from an eval fixture's optional `weather` block, so
    fixed weather can be injected into a run without any external API call.
    Returns None when the entry carries no weather (Stage 2 behaviour)."""
    weather = entry.get("weather")
    if not weather:
        return None
    return WeatherInput(
        temperature_c=weather.get("temperature_c"),
        condition=weather.get("condition"),
        location=weather.get("location"),
    )

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


def _pool_items(candidates):
    items = set()
    for candidate in candidates:
        items.update(candidate.items)
    return items


def _combo_in_pool(ids, candidates):
    needed = set(ids)
    return any(needed.issubset(set(candidate.items)) for candidate in candidates)


def _evaluate_check(hard_checks, ranked, candidates):
    """
    Evaluate an entry's hard_checks against the ranked output AND the candidate
    pool. Returns (status, reason, kind):
      - status: "pass" | "fail" | "inapplicable"
      - reason: human-readable explanation (None on pass)
      - kind: for inapplicable, "missing_item" or "missing_combination"; else None

    A check is inapplicable when the model was never given the chance: an item it
    references is in no candidate (missing_item), or the items all exist in the
    pool individually but never together in one candidate outfit
    (missing_combination). Inapplicable results are kept out of the pass rate so
    bounding the candidate pool does not silently look like failures - or passes.
    """
    pool = _pool_items(candidates)
    for key, value in hard_checks.items():
        if key == "top_rank_excludes":
            # Always applicable: this asserts the top pick omits certain items.
            present = [i for i in value if i in _top_items(ranked)]
            if present:
                return "fail", f"top-ranked outfit should exclude {present}", None
        elif key in ("combination_appears_ranked", "combination_not_top_rank"):
            missing = [i for i in value if i not in pool]
            if missing:
                return "inapplicable", f"item(s) not in candidate pool: {missing}", "missing_item"
            if not _combo_in_pool(value, candidates):
                return (
                    "inapplicable",
                    f"items present but no candidate outfit contains the combination {value}",
                    "missing_combination",
                )
            if key == "combination_appears_ranked":
                if not any(set(value).issubset(set(o.items)) for o in ranked):
                    return "fail", f"no ranked outfit contains the combination {value}", None
            else:
                if set(value).issubset(_top_items(ranked)):
                    return "fail", f"combination {value} should not be the top-ranked outfit", None
        elif key == "prefers_over":
            missing = [i for i in (value["higher"], value["lower"]) if i not in pool]
            if missing:
                return "inapplicable", f"item(s) not in candidate pool: {missing}", "missing_item"
            higher = _best_rank_containing(ranked, value["higher"])
            lower = _best_rank_containing(ranked, value["lower"])
            if higher is None:
                return "fail", f"preferred item {value['higher']} does not appear in any ranked outfit", None
            if lower is not None and not (higher < lower):
                return "fail", f"{value['higher']} should rank above {value['lower']} (ranks {higher} vs {lower})", None
        elif key == "reasoning_flags_gap":
            # Always applicable: this inspects the reasoning text.
            text = " ".join(outfit.reasoning.lower() for outfit in ranked)
            if value and not any(keyword in text for keyword in GAP_KEYWORDS):
                return "fail", "reasoning did not flag a wardrobe gap", None
        else:
            return "fail", f"unknown hard_check: {key}", None
    return "pass", None, None


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

    A ranking check is "inapplicable" when the model was never given the chance
    (an item or the required combination was not in the candidate pool). These
    are kept OUT of the pass rate, and applicable-coverage (applicable / total
    ranking checks) is reported so that bounding the candidate pool shows up as
    an evaluation-coverage problem even when nothing fails.

    Returns {passed, failed, inapplicable, behavioral_pending,
    constraint_violations, coverage}. Prints a two-level summary (overall, then
    per category).
    """
    with open(golden_set_path, encoding="utf-8") as handle:
        entries = yaml.safe_load(handle)
    with open(wardrobe_path, encoding="utf-8") as handle:
        wardrobe = [WardrobeItem(**item) for item in json.load(handle)]

    graph = build_graph()
    passed: list[str] = []
    failed: list[dict] = []
    inapplicable: list[dict] = []
    errored: list[dict] = []
    behavioral_pending: list[dict] = []
    constraint_violations: list[dict] = []
    # per-category tallies for ranking checks: {cat: {passed, failed, inapplicable}}
    by_category: dict[str, dict[str, int]] = {}

    for index, entry in enumerate(entries):
        if delay_seconds and index:
            time.sleep(delay_seconds)
        try:
            # Per-case isolation: a single bad generation (e.g. truncated JSON)
            # or transient error must not abort the whole pass.
            result = graph.invoke(
                {
                    "wardrobe": wardrobe,
                    "occasion": entry["occasion"],
                    "candidate_outfits": [],
                    "ranked_outfits": [],
                    # Fixed weather from the fixture, if any (no external API call).
                    "weather_input": _weather_input_from_entry(entry),
                }
            )
        except Exception as exc:
            errored.append({"id": entry["id"], "error": repr(exc)})
            print(f"  {entry['id']}: errored ({exc!r})")
            continue
        ranked = result["ranked_outfits"]
        candidates = result["candidate_outfits"]

        # Hard constraints: structural validity on every output, regardless of type.
        violations = validate_ranked_outfits(ranked, wardrobe)
        if violations:
            constraint_violations.append({"id": entry["id"], "violations": violations})

        if entry.get("type") == "behavioral" or entry.get("judged") is True:
            top_items = ranked[0].items if ranked else None
            top_reasoning = ranked[0].reasoning if ranked else None
            record = {
                "id": entry["id"],
                "occasion": entry["occasion"],
                "notes": entry.get("notes"),
                "judge_question": entry.get("judge_question"),
                "expectation": entry.get("hard_checks"),
                "top_outfit": top_items,
                "top_reasoning": top_reasoning,
            }
            # Only the rubric-bearing color cases are auto-judged; everything
            # else behavioral stays collected for manual review.
            rubric = entry.get("rubric")
            if rubric and top_items:
                names_by_id = {item.id: item.name for item in wardrobe}
                outfit_desc = ", ".join(names_by_id.get(i, i) for i in top_items)
                try:
                    verdict = judge_outfit(
                        outfit_desc, top_reasoning or "", entry["occasion"], rubric
                    )
                    record["verdict"] = verdict.model_dump()
                except Exception as error:
                    record["judge_error"] = str(error)
            behavioral_pending.append(record)
            continue

        status, reason, kind = _evaluate_check(entry["hard_checks"], ranked, candidates)
        category = entry.get("category", "uncategorized")
        tally = by_category.setdefault(
            category, {"passed": 0, "failed": 0, "inapplicable": 0}
        )
        if status == "pass":
            passed.append(entry["id"])
            tally["passed"] += 1
        elif status == "fail":
            failed.append({"id": entry["id"], "reason": reason})
            tally["failed"] += 1
        else:
            inapplicable.append({"id": entry["id"], "kind": kind, "reason": reason})
            tally["inapplicable"] += 1

    # Coverage: pass rate is over APPLICABLE checks only; applicable-coverage
    # exposes how much of the ranking suite actually ran against the bounded pool.
    applicable = len(passed) + len(failed)
    ranking_total = applicable + len(inapplicable)
    coverage = {
        "ranking_total": ranking_total,
        "applicable": applicable,
        "inapplicable": len(inapplicable),
        "passed": len(passed),
        "failed": len(failed),
        "pass_rate_applicable": (len(passed) / applicable) if applicable else None,
        "applicable_coverage": (applicable / ranking_total) if ranking_total else None,
        "by_category": by_category,
    }

    judged = [b for b in behavioral_pending if "verdict" in b]
    judged_passed = sum(1 for b in judged if b["verdict"]["passed"])
    pass_rate = f"{coverage['pass_rate_applicable']:.0%}" if applicable else "n/a"
    cov = f"{coverage['applicable_coverage']:.0%}" if ranking_total else "n/a"
    print("Ranking checks (inapplicable excluded from pass rate):")
    print(
        f"  overall: {len(passed)}/{applicable} applicable passed ({pass_rate}); "
        f"{len(inapplicable)} inapplicable; coverage {applicable}/{ranking_total} ({cov})"
    )
    for category in sorted(by_category):
        t = by_category[category]
        app = t["passed"] + t["failed"]
        total = app + t["inapplicable"]
        note = f" [{t['inapplicable']} inapplicable]" if t["inapplicable"] else ""
        print(f"  {category}: {t['passed']}/{app} passed, coverage {app}/{total}{note}")
    print(
        f"Behavioral: {len(behavioral_pending)} collected "
        f"({judged_passed}/{len(judged)} rubric-judged passed). "
        f"Structural violations: {len(constraint_violations)}. "
        f"Errored (could not evaluate): {len(errored)}."
    )
    return {
        "passed": passed,
        "failed": failed,
        "inapplicable": inapplicable,
        "errored": errored,
        "behavioral_pending": behavioral_pending,
        "constraint_violations": constraint_violations,
        "coverage": coverage,
    }


# Distinctive garment-type words used to check that an outfit's reasoning does
# not reference a garment the outfit does not contain (the g23 failure mode).
_GARMENT_WORDS = (
    "blazer",
    "cardigan",
    "trench",
    "coat",
    "rain jacket",
    "jacket",
    "scarf",
    "hoodie",
    "sweater",
    "turtleneck",
    "blouse",
    "tank",
    "tee",
    "shirt",
    "trousers",
    "jeans",
    "denim",
    "skirt",
)


def check_reasoning_garments(outfit_items, reasoning, wardrobe):
    """
    Return garment words mentioned in the reasoning that are NOT present in the
    recommended outfit but DO exist elsewhere in the wardrobe - i.e. the model
    talked about a garment it did not actually pick. Empty list means consistent.
    """
    by_id = {item.id: item.name.lower() for item in wardrobe}
    outfit_text = " ".join(by_id.get(i, "") for i in outfit_items)
    wardrobe_text = " ".join(by_id.values())
    text = (reasoning or "").lower()

    offenders = []
    for word in _GARMENT_WORDS:
        if word in text and word not in outfit_text and word in wardrobe_text:
            offenders.append(word)
    return offenders


def compare_pairwise(
    outfit_a, outfit_b, occasion, wardrobe, retrieved_guidance=None, weather_context=None
):
    """
    Compare the suitability of two specific outfits for one occasion (and optional
    weather) by asking the stylist to rank ONLY those two. This measures A-vs-B
    suitability directly; it does not touch the main 20-candidate top 5.

    When weather_context is given it is passed to both retrieval and the stylist,
    so a weather eval can hold the outfits and occasion fixed and vary only the
    weather. Returns {"winner", "ranked", "reasoning_offenders"} - offenders are
    garments named in the winning outfit's reasoning but not actually in it.
    """
    candidates = [OutfitCandidate(items=outfit_a), OutfitCandidate(items=outfit_b)]
    if retrieved_guidance is None:
        retrieved_guidance = retrieve_style_guidance(
            occasion, weather_context=weather_context
        )
    ranked = rank_outfits(
        candidates, wardrobe, occasion, retrieved_guidance, weather_context=weather_context
    )

    winner = None
    offenders = []
    if ranked:
        top = set(ranked[0].items)
        if top == set(outfit_a):
            winner = "A"
        elif top == set(outfit_b):
            winner = "B"
        offenders = check_reasoning_garments(
            ranked[0].items, ranked[0].reasoning, wardrobe
        )
    return {
        "winner": winner,
        "reasoning_offenders": offenders,
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


def _weather_context_from_block(block):
    if not block:
        return None
    return WeatherContext(
        temperature_c=block.get("temperature_c"),
        condition=block.get("condition"),
        sources={
            k: "user"
            for k in ("temperature_c", "condition")
            if block.get(k) is not None
        },
    )


def run_weather_pairwise_case(case, wardrobe) -> dict:
    """
    Hold outfits A/B and the occasion FIXED and vary only the weather. Each
    `conditions` entry supplies a weather block and the expected winner for that
    weather. Reports ranking success (did the winner match the expectation)
    SEPARATELY from garment-consistency (did the winning reasoning mention only
    garments it actually contains) - a flip for the wrong reason is not a pass.
    """
    outfit_a = case["outfit_a"]
    outfit_b = case["outfit_b"]
    conditions = []
    for cond in case["conditions"]:
        weather = _weather_context_from_block(cond.get("weather"))
        result = compare_pairwise(
            outfit_a, outfit_b, case["occasion"], wardrobe, weather_context=weather
        )
        conditions.append(
            {
                "label": cond.get("label"),
                "weather": cond.get("weather"),
                "expected_winner": cond.get("expected_winner"),
                "winner": result["winner"],
                "ranking_passed": result["winner"] == cond.get("expected_winner"),
                "reasoning_offenders": result["reasoning_offenders"],
                "ranked": result["ranked"],
            }
        )

    ranking_passed = all(c["ranking_passed"] for c in conditions)
    consistent = all(not c["reasoning_offenders"] for c in conditions)
    return {
        "id": case["id"],
        "occasion": case["occasion"],
        "question": case.get("question"),
        "ranking_passed": ranking_passed,  # did weather move the ranking as expected
        "garment_consistent": consistent,  # reported separately, not conflated
        "conditions": conditions,
    }


def run_weather_pairwise_set(
    pairwise_set_path, wardrobe_path, delay_seconds: float = 0
) -> list[dict]:
    """Run all weather pairwise cases, throttling between model calls. Prints a
    summary that keeps ranking success distinct from garment consistency."""
    with open(pairwise_set_path, encoding="utf-8") as handle:
        cases = yaml.safe_load(handle)
    with open(wardrobe_path, encoding="utf-8") as handle:
        wardrobe = [WardrobeItem(**item) for item in json.load(handle)]

    results = []
    first = True
    for case in cases:
        # delay between cases; compare_pairwise itself makes one call per condition
        if delay_seconds and not first:
            time.sleep(delay_seconds)
        first = False
        result = run_weather_pairwise_case(case, wardrobe)
        results.append(result)
        print(
            f"{case['id']}: ranking {'ok' if result['ranking_passed'] else 'FAIL'}, "
            f"garment-consistent {'ok' if result['garment_consistent'] else 'FAIL'}"
        )
    passed = sum(1 for r in results if r["ranking_passed"])
    consistent = sum(1 for r in results if r["garment_consistent"])
    print(
        f"weather pairwise: {passed}/{len(results)} ranking passed, "
        f"{consistent}/{len(results)} garment-consistent"
    )
    return results


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
