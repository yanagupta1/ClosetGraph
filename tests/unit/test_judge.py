import pytest

from closetgraph.judge import JudgeVerdict, judge_outfit


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


class FakeChat:
    def __init__(self, completions):
        self.completions = completions


class FakeGroq:
    completions = None

    def __init__(self):
        self.chat = FakeChat(self.completions)


def _args():
    return dict(
        outfit="navy long-sleeve top, camel trousers",
        reasoning="Neutrals with a warm camel anchor read intentional.",
        occasion="dinner date, upscale restaurant",
        rubric="Pass if the palette is intentional and elevated.",
    )


def test_judge_parses_successful_pass_verdict(monkeypatch):
    FakeGroq.completions = FakeCompletions(
        ['{"passed": true, "score": 4, "rationale": "Cohesive neutral palette."}']
    )
    monkeypatch.setattr("closetgraph.judge.Groq", FakeGroq)
    monkeypatch.setenv("JUDGE_MODEL", "test-judge")

    verdict = judge_outfit(**_args())

    assert isinstance(verdict, JudgeVerdict)
    assert verdict.passed is True
    assert verdict.score == 4
    assert FakeGroq().chat.completions is FakeGroq.completions  # sanity
    assert FakeGroq.completions.calls[0]["model"] == "test-judge"


def test_judge_parses_failed_verdict(monkeypatch):
    FakeGroq.completions = FakeCompletions(
        ['{"passed": false, "score": 2, "rationale": "Colours clash without logic."}']
    )
    monkeypatch.setattr("closetgraph.judge.Groq", FakeGroq)

    verdict = judge_outfit(**_args())

    assert verdict.passed is False
    assert verdict.score == 2


def test_judge_raises_on_persistently_malformed_response(monkeypatch):
    # Both the first call and the stricter retry return non-JSON -> propagate.
    FakeGroq.completions = FakeCompletions(["not json at all", "still not json"])
    monkeypatch.setattr("closetgraph.judge.Groq", FakeGroq)

    with pytest.raises(ValueError):
        judge_outfit(**_args())

    assert len(FakeGroq.completions.calls) == 2  # tried once, retried once
