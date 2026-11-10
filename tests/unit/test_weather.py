import json
from datetime import date, timedelta

import pytest

from closetgraph.daytime import TimeWindow
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


# --- time-aware forecast (network mocked) ---

def test_fetch_open_meteo_daily_forecast(monkeypatch):
    monkeypatch.setattr(
        "closetgraph.weather.urlopen",
        _fake_urlopen(
            geocode={"results": [{"latitude": 42.36, "longitude": -71.06}]},
            forecast={
                "daily": {
                    "temperature_2m_max": [12.0],
                    "temperature_2m_min": [4.0],
                    "weather_code": [61],
                }
            },
        ),
    )

    target = date.today() + timedelta(days=3)
    result = fetch_open_meteo("Boston", target_date=target)

    assert result["temperature_c"] == 8.0  # midpoint of high/low
    assert result["condition"] == "rainy"
    assert "high 12C / low 4C" in result["forecast_note"]


def test_fetch_open_meteo_hourly_window(monkeypatch):
    target = date.today() + timedelta(days=2)
    prefix = target.isoformat()
    hours = [16, 17, 18, 19, 20, 21, 22]
    monkeypatch.setattr(
        "closetgraph.weather.urlopen",
        _fake_urlopen(
            geocode={"results": [{"latitude": 42.36, "longitude": -71.06}]},
            forecast={
                "hourly": {
                    "time": [f"{prefix}T{h:02d}:00" for h in hours],
                    "temperature_2m": [10, 9, 8, 7, 6, 5, 4],
                    "weather_code": [61, 3, 3, 3, 3, 3, 0],
                }
            },
        ),
    )

    # Evening window 17-22 picks hours 17..21 -> temps 9,8,7,6,5 (avg 7.0).
    result = fetch_open_meteo(
        "Boston", target_date=target, time_window=TimeWindow(17, 22, "evening")
    )

    assert result["temperature_c"] == 7.0
    assert result["condition"] == "cloudy"  # dominant code 3
    assert "evening" in result["forecast_note"]


def test_fetch_open_meteo_date_beyond_window_raises(monkeypatch):
    monkeypatch.setattr(
        "closetgraph.weather.urlopen",
        _fake_urlopen(geocode={"results": [{"latitude": 1.0, "longitude": 2.0}]}),
    )

    with pytest.raises(ValueError, match="beyond the"):
        fetch_open_meteo("Boston", target_date=date.today() + timedelta(days=40))


def test_fetch_open_meteo_past_date_raises(monkeypatch):
    monkeypatch.setattr(
        "closetgraph.weather.urlopen",
        _fake_urlopen(geocode={"results": [{"latitude": 1.0, "longitude": 2.0}]}),
    )

    with pytest.raises(ValueError, match="in the past"):
        fetch_open_meteo("Boston", target_date=date.today() - timedelta(days=1))


def test_resolve_weather_threads_when_into_forecast():
    calls = []

    def fetcher(location, target_date=None, time_window=None):
        calls.append((location, target_date, time_window))
        return {
            "temperature_c": 8.0,
            "condition": "cloudy",
            "forecast_note": "forecast for 2026-11-14 (high 12C / low 4C)",
        }

    context, error = resolve_weather(
        WeatherInput(location="Boston", when="next Saturday"), fetcher=fetcher
    )

    assert error is None
    assert context.temperature_c == 8.0
    assert context.sources == {"temperature_c": "open-meteo", "condition": "open-meteo"}
    assert context.forecast_note.startswith("forecast for")
    # Date-only hint -> no hourly window threaded.
    assert calls[0][2] is None


def test_resolve_weather_threads_time_window_for_meal_hint():
    calls = []

    def fetcher(location, target_date=None, time_window=None):
        calls.append((location, target_date, time_window))
        return {"temperature_c": 6.0, "condition": "clear", "forecast_note": "note"}

    resolve_weather(
        WeatherInput(location="Boston", when="dinner next Saturday"), fetcher=fetcher
    )

    assert calls[0][2] is not None
    assert calls[0][2].label == "dinner"


def test_resolve_weather_unparseable_when_does_not_fetch():
    def boom(location, **kwargs):
        raise AssertionError("must not fetch when the date hint is uninterpretable")

    context, error = resolve_weather(
        WeatherInput(location="Boston", when="whenever-ish"), fetcher=boom
    )

    assert context is None  # no fabricated current-conditions substitute
    assert error is not None and "could not interpret" in error


def test_resolve_weather_when_without_location_is_surfaced():
    context, error = resolve_weather(WeatherInput(when="tomorrow evening"))

    assert context is None
    assert error is not None and "needs a location" in error


def test_resolve_weather_keeps_user_temp_when_forecast_fill_fails():
    def fetcher(location, target_date=None, time_window=None):
        raise RuntimeError("forecast down")

    context, error = resolve_weather(
        WeatherInput(temperature_c=5, location="Boston", when="tomorrow"),
        fetcher=fetcher,
    )

    assert context.temperature_c == 5  # user field preserved
    assert context.condition is None  # not fabricated
    assert error is not None and "forecast down" in error
