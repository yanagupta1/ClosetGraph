from closetgraph.supervisor import _llm_classify, supervise


def _never_called(occasion):
    raise AssertionError("classifier must not be called on a deterministic skip")


class FakeMessage:
    def __init__(self, content):
        self.content = content


class FakeChoice:
    def __init__(self, content):
        self.message = FakeMessage(content)


class FakeResponse:
    def __init__(self, content):
        self.choices = [FakeChoice(content)]


class FakeCompletions:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return FakeResponse(self.responses.pop(0))


class FakeGroq:
    completions = None

    def __init__(self):
        self.chat = type("Chat", (), {"completions": self.completions})()


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


def test_escalation_marker_without_attire_word_defers():
    # Covered word "cocktail" is present, but an escalation marker (masquerade/
    # ball) means we must not take the deterministic skip.
    def classifier(occasion):
        return {"research_needed": True, "research_query": "masquerade ball attire"}

    decision = supervise("cocktail at a masquerade ball", classifier=classifier)

    assert decision["research_needed"] is True


# --- _llm_classify: the default classifier (Groq mocked) ---

def test_llm_classify_parses_research_decision(monkeypatch):
    FakeGroq.completions = FakeCompletions(
        ['{"research_needed": true, "research_query": "Royal Ascot", '
         '"research_reason": "enclosure dress code is not general knowledge"}']
    )
    monkeypatch.setattr("closetgraph.supervisor.Groq", FakeGroq)
    monkeypatch.setenv("SUPERVISOR_MODEL", "test-supervisor")

    decision = _llm_classify("the Royal Enclosure at Royal Ascot")

    assert decision["research_needed"] is True
    assert decision["research_query"] == "Royal Ascot"
    assert FakeGroq.completions.calls[0]["model"] == "test-supervisor"


def test_llm_classify_drops_query_when_no_research_needed(monkeypatch):
    FakeGroq.completions = FakeCompletions(
        ['{"research_needed": false, "research_query": "office", '
         '"research_reason": "an office is well understood"}']
    )
    monkeypatch.setattr("closetgraph.supervisor.Groq", FakeGroq)

    decision = _llm_classify("the office")

    assert decision["research_needed"] is False
    assert decision["research_query"] is None  # query suppressed when not needed


def test_llm_classify_treats_null_query_string_as_none(monkeypatch):
    FakeGroq.completions = FakeCompletions(
        ['{"research_needed": true, "research_query": "null", "research_reason": "x"}']
    )
    monkeypatch.setattr("closetgraph.supervisor.Groq", FakeGroq)

    decision = _llm_classify("something ambiguous")

    assert decision["research_needed"] is True
    assert decision["research_query"] is None  # literal "null" normalized to None


def test_llm_classify_fails_safe_to_no_research_on_bad_response(monkeypatch):
    # A malformed (non-JSON) model response must not crash routing; it fails safe
    # to no-research so the pipeline continues.
    FakeGroq.completions = FakeCompletions(["not json at all"])
    monkeypatch.setattr("closetgraph.supervisor.Groq", FakeGroq)

    decision = _llm_classify("the Royal Enclosure at Royal Ascot")

    assert decision["research_needed"] is False
    assert decision["research_query"] is None
    assert "classifier unavailable" in decision["research_reason"]


def test_supervise_falls_through_to_llm_classify_by_default(monkeypatch):
    # No classifier injected and not a deterministic skip -> default _llm_classify.
    FakeGroq.completions = FakeCompletions(
        ['{"research_needed": true, "research_query": "Henley Royal Regatta", '
         '"research_reason": "stewards enclosure code"}']
    )
    monkeypatch.setattr("closetgraph.supervisor.Groq", FakeGroq)

    decision = supervise("the Stewards' Enclosure at Henley")

    assert decision["research_needed"] is True
    assert decision["research_query"] == "Henley Royal Regatta"
