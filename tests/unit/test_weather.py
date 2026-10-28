from closetgraph.models import WeatherInput
from closetgraph.weather import resolve_weather


def test_no_weather_input_resolves_to_nothing():
    context, error = resolve_weather(None)

    assert context is None
    assert error is None


def test_user_provided_weather_is_tagged_user_source():
    context, error = resolve_weather(
        WeatherInput(temperature_c=4, condition="cold and windy")
    )

    assert error is None
    assert context.source == "user"
    assert context.temperature_c == 4
    assert context.condition == "cold and windy"


def test_user_weather_takes_precedence_over_lookup():
    calls = []

    def fetcher(location):
        calls.append(location)
        return {"temperature_c": 99, "condition": "heatwave"}

    context, error = resolve_weather(
        WeatherInput(temperature_c=2, condition="snow", location="Oslo"),
        fetcher=fetcher,
    )

    assert context.source == "user"
    assert context.temperature_c == 2
    assert calls == []  # fetcher never consulted when the user supplied weather


def test_location_only_uses_fetcher_and_tags_open_meteo():
    context, error = resolve_weather(
        WeatherInput(location="Paris"),
        fetcher=lambda location: {"temperature_c": 15, "condition": "clear"},
    )

    assert error is None
    assert context.source == "open-meteo"
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
