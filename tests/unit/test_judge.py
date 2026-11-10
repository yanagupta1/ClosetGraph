import pytest

from closetgraph.judge import JudgeVerdict, judge_outfit, judge_routing_reason


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


def test_judge_raises_on_empty_response(monkeypatch):
    # An empty string both times -> the "empty" branch raises, not a crash.
    FakeGroq.completions = FakeCompletions(["", ""])
    monkeypatch.setattr("closetgraph.judge.Groq", FakeGroq)

    with pytest.raises(ValueError, match="empty"):
        judge_outfit(**_args())


def test_judge_strips_markdown_fenced_json(monkeypatch):
    FakeGroq.completions = FakeCompletions(
        ['```json\n{"passed": true, "score": 5, "rationale": "Clean."}\n```']
    )
    monkeypatch.setattr("closetgraph.judge.Groq", FakeGroq)

    verdict = judge_outfit(**_args())

    assert verdict.passed is True and verdict.score == 5


def test_judge_routing_reason_parses_verdict(monkeypatch):
    FakeGroq.completions = FakeCompletions(
        ['{"passed": true, "score": 4, "rationale": "Names the uncertain code."}']
    )
    monkeypatch.setattr("closetgraph.judge.Groq", FakeGroq)
    monkeypatch.setenv("JUDGE_MODEL", "test-judge")

    verdict = judge_routing_reason(
        "the Royal Enclosure at Royal Ascot",
        "research",
        "the enclosure dress code is not general knowledge",
    )

    assert isinstance(verdict, JudgeVerdict)
    assert verdict.passed is True
    assert FakeGroq.completions.calls[0]["model"] == "test-judge"


def test_judge_routing_reason_retries_after_bad_parse(monkeypatch):
    # First response is unparseable; the stricter retry returns valid JSON.
    FakeGroq.completions = FakeCompletions(
        ["not json", '{"passed": false, "score": 2, "rationale": "Circular reason."}']
    )
    monkeypatch.setattr("closetgraph.judge.Groq", FakeGroq)

    verdict = judge_routing_reason("office", "skip", "an office is obvious")

    assert verdict.passed is False
    assert len(FakeGroq.completions.calls) == 2  # retried once
    assert FakeGroq.completions.calls[1]["messages"][0]["role"] == "system"
