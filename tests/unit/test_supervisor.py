from closetgraph.supervisor import supervise


def _never_called(occasion):
    raise AssertionError("classifier must not be called on a deterministic skip")


def test_familiar_plain_occasion_skips_research_without_llm():
    decision = supervise("first client meeting, conservative office", classifier=_never_called)

    assert decision["research_needed"] is False
    assert decision["research_query"] is None
    assert decision["research_reason"]  # a reason is recorded for explainability


def test_unfamiliar_venue_without_dress_code_is_not_researched():
    # A named cafe: style guidance is enough; the classifier should decline research.
    def classifier(occasion):
        return {
            "research_needed": False,
            "research_query": None,
            "research_reason": "a cafe has no specific dress code to look up",
        }

    decision = supervise("coffee at Blumen Cafe", classifier=classifier)

    assert decision["research_needed"] is False
    assert "cafe" in decision["research_reason"]


def test_escalation_marker_defers_to_classifier():
    seen = {}

    def classifier(occasion):
        seen["occasion"] = occasion
        return {"research_needed": True, "research_query": "charity gala dress code"}

    decision = supervise("cocktail attire for a charity gala", classifier=classifier)

    # "gala" is an escalation marker, so even with "cocktail" present we do not skip.
    assert seen["occasion"] == "cocktail attire for a charity gala"
    assert decision["research_needed"] is True
    assert decision["research_query"] == "charity gala dress code"


def test_named_venue_defers_to_classifier():
    called = {"n": 0}

    def classifier(occasion):
        called["n"] += 1
        return {"research_needed": True, "research_query": "the Met Gala"}

    decision = supervise("cocktail party at the Met", classifier=classifier)

    assert called["n"] == 1  # capitalized "Met" is a research signal, not skipped
    assert decision["research_needed"] is True


def test_unfamiliar_occasion_no_familiar_anchor_defers_to_classifier():
    def classifier(occasion):
        return {"research_needed": True, "research_query": "Hunt Ball attire"}

    decision = supervise("what to wear to a hunt ball", classifier=classifier)

    assert decision["research_needed"] is True
    assert decision["research_query"] == "Hunt Ball attire"
