"""Integration test: weather RAG retrieves the right chunk per condition.

Builds a small Chroma index from the real style-guide corpus into a temp dir
(local, no network) and checks that cold/hot/rainy/windy queries each surface
their matching weather section, and that a no-weather query pulls no weather
chunk at all.
"""
import chromadb
import pytest

from closetgraph.indexing import chunk_markdown_file
from closetgraph.models import WeatherContext


@pytest.fixture
def weather_collection(tmp_path):
    client = chromadb.PersistentClient(path=str(tmp_path / "chroma"))
    collection = client.get_or_create_collection("style_guide")
    for category, path in [
        ("occasion", "style_guide/occasion.md"),
        ("weather", "style_guide/weather.md"),
    ]:
        chunks = chunk_markdown_file(path, category)
        collection.add(
            documents=[c["text"] for c in chunks],
            metadatas=[{"category": c["category"], "heading": c["heading"]} for c in chunks],
            ids=[c["id"] for c in chunks],
        )
    return tmp_path


def _retrieve(persist_dir, occasion, weather_context=None):
    from closetgraph.retrieval import retrieve_style_guidance

    # point retrieval at the temp index; only occasion+weather categories exist,
    # missing categories simply return nothing.
    return retrieve_style_guidance(
        occasion,
        persist_path=str(persist_dir / "chroma"),
        weather_context=weather_context,
    )


@pytest.mark.parametrize(
    ("condition", "temperature_c", "expected_heading"),
    [
        ("cold", 1, "Cold Conditions"),
        ("hot", 33, "Warm Conditions"),
        ("rainy", 10, "Precipitation"),
        ("windy", 14, "Wind"),
    ],
)
def test_weather_retrieval_matches_condition(
    weather_collection, condition, temperature_c, expected_heading
):
    ctx = WeatherContext(
        temperature_c=temperature_c,
        condition=condition,
        sources={"condition": "user"},
    )
    chunks = _retrieve(weather_collection, "a day out", weather_context=ctx)

    weather_chunks = [c for c in chunks if c.lstrip().startswith("## ")]
    joined = "\n".join(chunks)
    assert expected_heading in joined, (
        f"{condition!r} should retrieve {expected_heading!r}; got {weather_chunks}"
    )


def test_no_weather_retrieves_no_weather_chunk(weather_collection):
    chunks = _retrieve(weather_collection, "a day out", weather_context=None)

    for heading in ("Cold Conditions", "Warm Conditions", "Precipitation", "Wind"):
        assert heading not in "\n".join(chunks)
