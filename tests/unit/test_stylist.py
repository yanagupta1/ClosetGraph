import json

import pytest

from closetgraph.models import OutfitCandidate, RankedOutfit
from closetgraph.stylist import (
    _extract_json_array,
    _fallback_rank_outfits,
    _parse_ranked_outfits,
    _resolve_candidates,
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


def test_resolve_candidates_replaces_ids_with_full_items(sample_wardrobe):
    candidates = [OutfitCandidate(items=["t2", "b1", "l1"])]

    resolved = _resolve_candidates(candidates, sample_wardrobe)

    assert resolved == [
        {
            "items": [
                {"id": "t2", "name": "white silk blouse", "category": "top"},
                {"id": "b1", "name": "black tailored trousers", "category": "bottom"},
                {"id": "l1", "name": "black structured blazer", "category": "layering"},
            ]
        }
    ]


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


def test_rank_outfits_falls_back_when_groq_call_fails(monkeypatch, sample_wardrobe):
    completions = FakeCompletions([RuntimeError("api down")])
    FakeGroq.completions = completions
    monkeypatch.setattr("closetgraph.stylist.Groq", FakeGroq)

    ranked = rank_outfits(
        [OutfitCandidate(items=["t1", "b1", "l1"])],
        sample_wardrobe,
        "first client meeting, conservative office",
    )

    assert len(ranked) == 1
    assert ranked[0].items == ["t1", "b1", "l1"]


def test_fallback_rank_outfits_prefers_polished_items_for_professional_context(sample_wardrobe):
    candidates = [
        OutfitCandidate(items=["t1", "b2"]),
        OutfitCandidate(items=["t2", "b1", "l1"]),
    ]

    ranked = _fallback_rank_outfits(
        candidates,
        sample_wardrobe,
        "first client meeting, conservative office",
    )

    assert ranked[0].items == ["t2", "b1", "l1"]
