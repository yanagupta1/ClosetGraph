import json

import pytest
from groq import RateLimitError

from closetgraph.models import OutfitCandidate, RankedOutfit, WeatherContext
from closetgraph.stylist import (
    _create_with_backoff,
    _extract_json_array,
    _format_candidates,
    _format_weather,
    _parse_ranked_outfits,
    rank_outfits,
)


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
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return FakeResponse(response)


class FakeChat:
    def __init__(self, completions):
        self.completions = completions


class FakeGroq:
    completions = None

    def __init__(self):
        self.chat = FakeChat(self.completions)


def test_format_candidates_produces_compact_lines(sample_wardrobe):
    candidates = [OutfitCandidate(items=["t2", "b1", "l1"])]

    formatted = _format_candidates(candidates, sample_wardrobe)

    assert formatted == (
        "1. t2 (white silk blouse, top) + b1 (black tailored trousers, bottom) "
        "+ l1 (black structured blazer, layering)"
    )


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ('[{"rank": 1, "items": ["t1"], "reasoning": "ok"}]', '[{"rank": 1, "items": ["t1"], "reasoning": "ok"}]'),
        ('```json\n[{"rank": 1, "items": ["t1"], "reasoning": "ok"}]\n```', '[{"rank": 1, "items": ["t1"], "reasoning": "ok"}]'),
        ('{"ranked_outfits": [{"rank": 1, "items": ["t1"], "reasoning": "ok"}]}', '[{"rank": 1, "items": ["t1"], "reasoning": "ok"}]'),
        ('Here is JSON: [{"rank": 1, "items": ["t1"], "reasoning": "ok"}]', '[{"rank": 1, "items": ["t1"], "reasoning": "ok"}]'),
    ],
)
def test_extract_json_array_accepts_common_model_response_shapes(content, expected):
    assert json.loads(_extract_json_array(content)) == json.loads(expected)


def test_extract_json_array_rejects_empty_response():
    with pytest.raises(ValueError, match="empty"):
        _extract_json_array("")


def test_parse_ranked_outfits_sorts_by_rank():
    parsed = _parse_ranked_outfits(
        json.dumps(
            [
                {"rank": 2, "items": ["t1", "b1"], "reasoning": "second"},
                {"rank": 1, "items": ["t2", "b1"], "reasoning": "first"},
            ]
        )
    )

    assert [outfit.rank for outfit in parsed] == [1, 2]
    assert all(isinstance(outfit, RankedOutfit) for outfit in parsed)


def test_rank_outfits_uses_groq_response(monkeypatch, sample_wardrobe):
    completions = FakeCompletions(
        [
            json.dumps(
                [
                    {
                        "rank": 1,
                        "items": ["t2", "b1", "l1"],
                        "reasoning": "The blouse, trousers, and blazer are polished.",
                    }
                ]
            )
        ]
    )
    FakeGroq.completions = completions
    monkeypatch.setattr("closetgraph.stylist.Groq", FakeGroq)
    monkeypatch.setenv("GROQ_MODEL", "test-model")

    ranked = rank_outfits(
        [OutfitCandidate(items=["t2", "b1", "l1"])],
        sample_wardrobe,
        "first client meeting",
    )

    assert ranked[0].items == ["t2", "b1", "l1"]
    assert completions.calls[0]["model"] == "test-model"
    assert "white silk blouse" in completions.calls[0]["messages"][0]["content"]
    # No weather provided -> prompt says so, not a fabricated condition.
    assert "Weather: none provided" in completions.calls[0]["messages"][0]["content"]


def test_rank_outfits_includes_weather_in_prompt_when_provided(monkeypatch, sample_wardrobe):
    from closetgraph.models import WeatherContext

    completions = FakeCompletions(
        [json.dumps([{"rank": 1, "items": ["t2", "b1"], "reasoning": "ok"}])]
    )
    FakeGroq.completions = completions
    monkeypatch.setattr("closetgraph.stylist.Groq", FakeGroq)

    rank_outfits(
        [OutfitCandidate(items=["t2", "b1"])],
        sample_wardrobe,
        "client meeting",
        weather_context=WeatherContext(
            temperature_c=3, condition="rainy", sources={"temperature_c": "user"}
        ),
    )

    prompt = completions.calls[0]["messages"][0]["content"]
    assert "3 degrees celsius" in prompt
    assert "rainy" in prompt


def test_rank_outfits_includes_research_in_prompt_when_provided(monkeypatch, sample_wardrobe):
    from closetgraph.models import ResearchContext

    completions = FakeCompletions(
        [json.dumps([{"rank": 1, "items": ["t2", "b1"], "reasoning": "ok"}])]
    )
    FakeGroq.completions = completions
    monkeypatch.setattr("closetgraph.stylist.Groq", FakeGroq)

    rank_outfits(
        [OutfitCandidate(items=["t2", "b1"])],
        sample_wardrobe,
        "hunt ball",
        research_context=ResearchContext(
            query="hunt ball attire",
            summary="Formal evening dress is expected.",
            inferred_formality="formal",
        ),
    )

    prompt = completions.calls[0]["messages"][0]["content"]
    assert "Formal evening dress is expected." in prompt
    assert "expected formality: formal" in prompt


def test_rank_outfits_research_none_marker_when_absent(monkeypatch, sample_wardrobe):
    completions = FakeCompletions(
        [json.dumps([{"rank": 1, "items": ["t2", "b1"], "reasoning": "ok"}])]
    )
    FakeGroq.completions = completions
    monkeypatch.setattr("closetgraph.stylist.Groq", FakeGroq)

    rank_outfits([OutfitCandidate(items=["t2", "b1"])], sample_wardrobe, "office")

    assert "Researched context for this occasion: none" in completions.calls[0]["messages"][0]["content"]


def test_rank_outfits_retries_after_invalid_json(monkeypatch, sample_wardrobe):
    completions = FakeCompletions(
        [
            "not json",
            json.dumps(
                [
                    {
                        "rank": 1,
                        "items": ["t1", "b1"],
                        "reasoning": "The outfit is clean and simple.",
                    }
                ]
            ),
        ]
    )
    FakeGroq.completions = completions
    monkeypatch.setattr("closetgraph.stylist.Groq", FakeGroq)

    ranked = rank_outfits(
        [OutfitCandidate(items=["t1", "b1"])],
        sample_wardrobe,
        "casual brunch",
    )

    assert ranked[0].rank == 1
    assert len(completions.calls) == 2
    assert completions.calls[1]["messages"][0]["role"] == "system"


def test_rank_outfits_raises_on_persistent_failure(monkeypatch, sample_wardrobe):
    # The Groq call itself fails: the real error propagates, no fallback.
    FakeGroq.completions = FakeCompletions([RuntimeError("api down")])
    monkeypatch.setattr("closetgraph.stylist.Groq", FakeGroq)

    with pytest.raises(RuntimeError, match="api down"):
        rank_outfits(
            [OutfitCandidate(items=["t1", "b1", "l1"])],
            sample_wardrobe,
            "first client meeting, conservative office",
        )

    # Both parse attempts fail: the parse ValueError propagates, no fallback.
    FakeGroq.completions = FakeCompletions(["not json", "still not json"])
    monkeypatch.setattr("closetgraph.stylist.Groq", FakeGroq)

    with pytest.raises(ValueError):
        rank_outfits(
            [OutfitCandidate(items=["t1", "b1", "l1"])],
            sample_wardrobe,
            "first client meeting, conservative office",
        )


class FakeUsage:
    def __init__(self, prompt_tokens, completion_tokens):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class FakeCompletionsWithUsage:
    def __init__(self, content):
        self.content = content
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        response = FakeResponse(self.content)
        response.usage = FakeUsage(prompt_tokens=10, completion_tokens=5)
        return response


def test_langfuse_client_initializes():
    from langfuse import get_client

    assert get_client() is not None


# --- _create_with_backoff: rate-limit retry behavior ---

def _rate_limit_error(message):
    # Construct a RateLimitError without its network-bound __init__ so str(err)
    # carries the message the backoff parses.
    err = RateLimitError.__new__(RateLimitError)
    err.args = (message,)
    return err


class _BackoffCompletions:
    def __init__(self, fail_times, message):
        self.fail_times = fail_times
        self.message = message
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise _rate_limit_error(self.message)
        return "OK"


class _BackoffClient:
    def __init__(self, completions):
        self.chat = type("Chat", (), {"completions": completions})()


def test_create_with_backoff_waits_then_succeeds(monkeypatch):
    slept = []
    monkeypatch.setattr("closetgraph.stylist.time.sleep", lambda s: slept.append(s))
    comp = _BackoffCompletions(fail_times=1, message="Rate limit reached, try again in 1.5s")
    client = _BackoffClient(comp)

    result = _create_with_backoff(client, model="m")

    assert result == "OK"
    assert comp.calls == 2  # failed once, retried once
    assert slept == [3.5]  # parsed 1.5s + 2s buffer, under the 65s cap


def test_create_with_backoff_uses_default_wait_when_no_hint(monkeypatch):
    slept = []
    monkeypatch.setattr("closetgraph.stylist.time.sleep", lambda s: slept.append(s))
    comp = _BackoffCompletions(fail_times=1, message="rate limited, slow down")
    client = _BackoffClient(comp)

    _create_with_backoff(client, model="m")

    assert slept == [20.0]  # no "try again in Xs" -> conservative default


def test_create_with_backoff_reraises_after_exhausting_retries(monkeypatch):
    monkeypatch.setattr("closetgraph.stylist.time.sleep", lambda s: None)
    comp = _BackoffCompletions(fail_times=99, message="try again in 0.1s")
    client = _BackoffClient(comp)

    with pytest.raises(RateLimitError):
        _create_with_backoff(client, max_retries=2, model="m")

    assert comp.calls == 3  # initial try + 2 retries, then propagate


def test_parse_ranked_outfits_rejects_non_list_payload():
    # A JSON object under a known key that is not a list -> explicit error.
    with pytest.raises(ValueError, match="Expected a JSON array"):
        _parse_ranked_outfits('{"ranked_outfits": {"rank": 1}}')


def test_format_weather_appends_forecast_note():
    wc = WeatherContext(
        temperature_c=5,
        condition="clear",
        sources={"temperature_c": "open-meteo", "condition": "open-meteo"},
        forecast_note="forecast for 2026-11-14 evening (18:00-22:00, avg 5C)",
    )

    line = _format_weather(wc)

    assert line.startswith("5 degrees celsius, clear")
    assert "(forecast for 2026-11-14 evening" in line


def test_rank_outfits_logs_generation_usage_without_breaking(monkeypatch, sample_wardrobe):
    # Observed (@observe) call plus the Langfuse generation.update usage path
    # should run end to end on a mocked Groq response carrying .usage.
    content = json.dumps(
        [{"rank": 1, "items": ["t2", "b1", "l1"], "reasoning": "polished"}]
    )
    FakeGroq.completions = FakeCompletionsWithUsage(content)
    monkeypatch.setattr("closetgraph.stylist.Groq", FakeGroq)

    ranked = rank_outfits(
        [OutfitCandidate(items=["t2", "b1", "l1"])],
        sample_wardrobe,
        "first client meeting",
    )

    assert ranked[0].items == ["t2", "b1", "l1"]
