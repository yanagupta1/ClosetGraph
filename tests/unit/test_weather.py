import json

import pytest

from closetgraph.models import WeatherInput
from closetgraph.weather import fetch_open_meteo, resolve_weather


class FakeHTTPResponse:
    def __init__(self, payload):
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return json.dumps(self._payload).encode("utf-8")


def _fake_urlopen(geocode=None, forecast=None, raise_on=None):
    """Build a urlopen stub that returns geocode/forecast payloads by URL."""

    def urlopen(url, timeout=None):
        if raise_on and raise_on in url:
            raise OSError("network down")
        if "geocoding-api" in url:
            return FakeHTTPResponse(geocode)
        return FakeHTTPResponse(forecast)

    return urlopen


def test_no_weather_input_resolves_to_nothing():
    context, error = resolve_weather(None)

    assert context is None
    assert error is None


def test_user_provided_weather_is_tagged_user_source():
    context, error = resolve_weather(
        WeatherInput(temperature_c=4, condition="cold and windy")
    )

    assert error is None
    assert context.sources == {"temperature_c": "user", "condition": "user"}
    assert context.temperature_c == 4
    assert context.condition == "cold and windy"


def test_complete_user_weather_takes_precedence_no_lookup():
    calls = []

    def fetcher(location):
        calls.append(location)
        return {"temperature_c": 99, "condition": "heatwave"}

    context, error = resolve_weather(
        WeatherInput(temperature_c=2, condition="snow", location="Oslo"),
        fetcher=fetcher,
    )

    assert context.sources == {"temperature_c": "user", "condition": "user"}
    assert context.temperature_c == 2
    assert calls == []  # nothing missing -> fetcher never consulted


def test_partial_user_input_fetches_only_missing_field(monkeypatch):
    # User gives temperature but no condition -> keep the temp, fetch condition.
    calls = []

    def fetcher(location):
        calls.append(location)
        return {"temperature_c": 99, "condition": "snowy"}

    context, error = resolve_weather(
        WeatherInput(temperature_c=3, location="NYC"), fetcher=fetcher
    )

    assert error is None
    assert context.temperature_c == 3  # user value preserved, not overwritten by 99
    assert context.condition == "snowy"  # gap filled by lookup
    assert context.sources == {"temperature_c": "user", "condition": "open-meteo"}
    assert calls == ["NYC"]


def test_partial_input_without_location_keeps_user_field_no_error():
    # Temp only, no location -> cannot fill condition, but that is not an error.
    context, error = resolve_weather(WeatherInput(temperature_c=3))

    assert error is None
    assert context.temperature_c == 3
    assert context.condition is None
    assert context.sources == {"temperature_c": "user"}


def test_location_only_uses_fetcher_and_tags_open_meteo():
    context, error = resolve_weather(
        WeatherInput(location="Paris"),
        fetcher=lambda location: {"temperature_c": 15, "condition": "clear"},
    )

    assert error is None
    assert context.sources == {"temperature_c": "open-meteo", "condition": "open-meteo"}
    assert context.temperature_c == 15


def test_failed_lookup_does_not_fabricate_and_is_observable():
    def fetcher(location):
        raise RuntimeError("network down")

    context, error = resolve_weather(WeatherInput(location="Paris"), fetcher=fetcher)

    assert context is None  # no fabricated weather
    assert error is not None and "network down" in error


def test_location_without_fetcher_reports_error_not_fabrication():
    context, error = resolve_weather(WeatherInput(location="Paris"))

    assert context is None
    assert error is not None


# --- fetch_open_meteo (network mocked) ---

def test_fetch_open_meteo_success(monkeypatch):
    monkeypatch.setattr(
        "closetgraph.weather.urlopen",
        _fake_urlopen(
            geocode={"results": [{"latitude": 48.85, "longitude": 2.35}]},
            forecast={"current": {"temperature_2m": 7.0, "weather_code": 61}},
        ),
    )

    result = fetch_open_meteo("Paris")

    assert result == {"temperature_c": 7.0, "condition": "rainy"}


def test_fetch_open_meteo_unknown_location_raises(monkeypatch):
    monkeypatch.setattr(
        "closetgraph.weather.urlopen", _fake_urlopen(geocode={"results": []})
    )

    with pytest.raises(ValueError, match="no geocoding match"):
        fetch_open_meteo("Nowherecity")


def test_fetch_open_meteo_network_error_propagates(monkeypatch):
    monkeypatch.setattr(
        "closetgraph.weather.urlopen", _fake_urlopen(raise_on="geocoding-api")
    )

    with pytest.raises(OSError):
        fetch_open_meteo("Paris")


def test_fetch_open_meteo_malformed_forecast_raises(monkeypatch):
    monkeypatch.setattr(
        "closetgraph.weather.urlopen",
        _fake_urlopen(
            geocode={"results": [{"latitude": 1.0, "longitude": 2.0}]},
            forecast={"current": {}},  # missing temperature_2m
        ),
    )

    with pytest.raises(ValueError, match="missing current temperature"):
        fetch_open_meteo("Paris")


def test_resolve_weather_surfaces_fetch_failure_through_error(monkeypatch):
    monkeypatch.setattr(
        "closetgraph.weather.urlopen", _fake_urlopen(raise_on="geocoding-api")
    )

    context, error = resolve_weather(WeatherInput(location="Paris"), fetcher=fetch_open_meteo)

    assert context is None  # nothing user-provided, nothing fetched -> no context
    assert error is not None and "network down" in error


def test_failed_fill_keeps_user_field_and_surfaces_error(monkeypatch):
    # User gave temp, condition lookup fails -> keep temp, condition None, error set.
    monkeypatch.setattr(
        "closetgraph.weather.urlopen", _fake_urlopen(raise_on="geocoding-api")
    )

    context, error = resolve_weather(
        WeatherInput(temperature_c=3, location="Paris"), fetcher=fetch_open_meteo
    )

    assert context is not None
    assert context.temperature_c == 3  # user field not lost on fill failure
    assert context.condition is None  # not fabricated
    assert context.sources == {"temperature_c": "user"}
    assert error is not None and "network down" in error


def test_complete_user_weather_never_hits_network(monkeypatch):
    def boom(url, timeout=None):
        raise AssertionError("network must not be called when user weather is complete")

    monkeypatch.setattr("closetgraph.weather.urlopen", boom)

    context, error = resolve_weather(
        WeatherInput(temperature_c=3, condition="snowy", location="Oslo"),
        fetcher=fetch_open_meteo,
    )

    assert context.sources == {"temperature_c": "user", "condition": "user"}
    assert error is None
