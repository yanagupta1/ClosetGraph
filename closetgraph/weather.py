"""Weather resolution: turn an optional WeatherInput into a resolved
WeatherContext (or nothing), with clear provenance and graceful failure.

The Open-Meteo fetch (fetch_open_meteo) is keyless and uses only the stdlib. It
is injected into resolve_weather as a `fetcher` callable so the resolution logic
stays pure and testable without any external API call.

Time awareness: fetch_open_meteo resolves CURRENT conditions by default, but
when a target_date (and optional hourly time_window) is supplied it reads
Open-Meteo's forecast for that date instead. A date outside the free forecast
window, or in the past, raises rather than silently substituting current
conditions, so a future request is never answered with today's weather.
"""
import json
import logging
from collections import Counter
from datetime import date
from urllib.parse import urlencode
from urllib.request import urlopen

from closetgraph.daytime import TimeWindow, parse_when
from closetgraph.models import WeatherContext, WeatherInput

logger = logging.getLogger(__name__)

# Open-Meteo's free forecast reaches 16 days out (day index 0..15 from today).
_FORECAST_MAX_DAYS = 15

# A fetcher takes a location string and returns {"temperature_c": float|None,
# "condition": str|None}, or raises / returns None if it cannot resolve.
Fetcher = "Callable[[str], dict | None]"

_GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
_HTTP_TIMEOUT = 10

# WMO weather codes -> a coarse condition word the stylist/corpus can reason over.
_WMO_CONDITIONS = {
    0: "clear",
    1: "mostly clear",
    2: "partly cloudy",
    3: "cloudy",
    45: "foggy",
    48: "foggy",
    51: "drizzle",
    53: "drizzle",
    55: "drizzle",
    61: "rainy",
    63: "rainy",
    65: "rainy",
    66: "rainy",
    67: "rainy",
    71: "snowy",
    73: "snowy",
    75: "snowy",
    77: "snowy",
    80: "rainy",
    81: "rainy",
    82: "rainy",
    85: "snowy",
    86: "snowy",
    95: "thunderstorm",
    96: "thunderstorm",
    99: "thunderstorm",
}


def _get_json(url):
    with urlopen(url, timeout=_HTTP_TIMEOUT) as response:
        return json.loads(response.read().decode("utf-8"))


def _geocode(location: str) -> tuple[float, float]:
    geo = _get_json(f"{_GEOCODE_URL}?{urlencode({'name': location, 'count': 1})}")
    results = geo.get("results") or []
    if not results:
        raise ValueError(f"no geocoding match for location {location!r}")
    match = results[0]
    if "latitude" not in match or "longitude" not in match:
        raise ValueError(f"geocoding result missing coordinates for {location!r}")
    return match["latitude"], match["longitude"]


def _fetch_current(lat: float, lon: float) -> dict:
    params = {"latitude": lat, "longitude": lon, "current": "temperature_2m,weather_code"}
    data = _get_json(f"{_FORECAST_URL}?{urlencode(params)}")
    current = data.get("current") or {}
    if "temperature_2m" not in current:
        raise ValueError("forecast response missing current temperature")
    return {
        "temperature_c": current["temperature_2m"],
        "condition": _WMO_CONDITIONS.get(current.get("weather_code"), "unknown"),
    }


def _fetch_daily(lat: float, lon: float, target_date: date) -> dict:
    iso = target_date.isoformat()
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "temperature_2m_max,temperature_2m_min,weather_code",
        "start_date": iso,
        "end_date": iso,
        "timezone": "auto",
    }
    data = _get_json(f"{_FORECAST_URL}?{urlencode(params)}")
    daily = data.get("daily") or {}
    try:
        tmax = daily["temperature_2m_max"][0]
        tmin = daily["temperature_2m_min"][0]
        code = daily["weather_code"][0]
    except (KeyError, IndexError):
        raise ValueError(f"daily forecast missing data for {iso}")
    return {
        "temperature_c": round((tmax + tmin) / 2, 1),
        "condition": _WMO_CONDITIONS.get(code, "unknown"),
        "forecast_note": f"forecast for {iso} (high {tmax:g}C / low {tmin:g}C)",
    }


def _fetch_hourly(lat: float, lon: float, target_date: date, window: TimeWindow) -> dict:
    iso = target_date.isoformat()
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,weather_code",
        "start_date": iso,
        "end_date": iso,
        "timezone": "auto",
    }
    data = _get_json(f"{_FORECAST_URL}?{urlencode(params)}")
    hourly = data.get("hourly") or {}
    times = hourly.get("time") or []
    temps = hourly.get("temperature_2m") or []
    codes = hourly.get("weather_code") or []
    picked_temps, picked_codes = [], []
    for stamp, temp, code in zip(times, temps, codes):
        hour = int(stamp[11:13])  # ISO "2026-11-14T18:00"
        if window.start_hour <= hour < window.end_hour and temp is not None:
            picked_temps.append(temp)
            picked_codes.append(code)
    if not picked_temps:
        raise ValueError(f"no hourly forecast in window for {iso} {window.label}")
    avg = round(sum(picked_temps) / len(picked_temps), 1)
    dominant = Counter(picked_codes).most_common(1)[0][0]
    return {
        "temperature_c": avg,
        "condition": _WMO_CONDITIONS.get(dominant, "unknown"),
        "forecast_note": (
            f"forecast for {iso} {window.label} "
            f"({window.start_hour:02d}:00-{window.end_hour:02d}:00, avg {avg:g}C)"
        ),
    }


def fetch_open_meteo(
    location: str, target_date: date | None = None, time_window: TimeWindow | None = None
) -> dict:
    """
    Resolve weather for a location via Open-Meteo (keyless, stdlib). Returns
    {"temperature_c": float, "condition": str, optional "forecast_note": str}.

    - No target_date: CURRENT conditions (backward-compatible default).
    - target_date, no time_window: the DAILY forecast for that date (temperature
      reported as the day's midpoint, with the high/low in forecast_note).
    - target_date + time_window: the HOURLY forecast averaged over the window.

    Raises on an unknown location, a date in the past or beyond the forecast
    window, a network error, or a malformed response, so resolve_weather can
    surface the failure through weather_error. Never invents a location,
    fabricates conditions, or substitutes current weather for a future date.
    """
    lat, lon = _geocode(location)
    if target_date is None:
        return _fetch_current(lat, lon)

    days_out = (target_date - date.today()).days
    if days_out < 0:
        raise ValueError(f"{target_date.isoformat()} is in the past; no forecast available")
    if days_out > _FORECAST_MAX_DAYS:
        raise ValueError(
            f"{target_date.isoformat()} is beyond the "
            f"{_FORECAST_MAX_DAYS + 1}-day forecast window; no forecast available"
        )
    if time_window is None:
        return _fetch_daily(lat, lon, target_date)
    return _fetch_hourly(lat, lon, target_date, time_window)


def resolve_weather(weather_input: WeatherInput | None, fetcher=None):
    """
    Resolve weather for a request, returning (context, error):

      - context: a WeatherContext when any field was resolved, else None.
      - error:   a short string when a lookup was attempted and failed, else None.

    Per-field merge, not all-or-nothing:
      - No input, or nothing usable -> (None, None). Not an error; the request
        simply carries no weather (Stage 2 behaviour).
      - User-provided fields always win and are tagged "user". The fetcher is
        only consulted to fill fields the user left out (and only when a
        location is available), so e.g. "3C in NYC, no condition" keeps the
        user's temperature and fetches just the condition.
      - A lookup is only attempted when a field is missing AND a location is
        given. On failure we never fabricate: any user-provided fields are kept,
        the missing ones stay None, and the failure is surfaced via error.
      - A `when` hint ("next Friday", "tomorrow evening") is parsed into a
        forecast date and optional hourly window and threaded to the fetcher. An
        unrecognized hint is reported via error and never silently resolved to
        current conditions. A hint with no location to forecast is likewise
        surfaced, not guessed.
    """
    if weather_input is None:
        return None, None

    temperature_c = weather_input.temperature_c
    condition = weather_input.condition
    sources: dict[str, str] = {}
    if temperature_c is not None:
        sources["temperature_c"] = "user"
    if condition is not None:
        sources["condition"] = "user"

    error = None
    forecast_note = None
    parsed = parse_when(weather_input.when) if weather_input.when else None
    when_unparsed = bool(weather_input.when) and parsed is None

    missing = temperature_c is None or condition is None
    if missing and when_unparsed:
        error = (
            f"could not interpret date/time hint {weather_input.when!r}; "
            "no forecast fetched"
        )
        logger.warning(error)
    elif missing and parsed and not weather_input.location:
        error = (
            f"date/time hint {weather_input.when!r} needs a location to forecast; "
            "none given"
        )
        logger.warning(error)
    elif missing and weather_input.location:
        if fetcher is None:
            error = f"no weather fetcher available to resolve {weather_input.location!r}"
            logger.warning(error)
        else:
            fetch_kwargs = {}
            if parsed:
                fetch_kwargs = {
                    "target_date": parsed.target_date,
                    "time_window": parsed.time_window,
                }
            try:
                fetched = fetcher(weather_input.location, **fetch_kwargs)
            except Exception as exc:  # never fabricate weather on failure
                error = f"weather lookup failed for {weather_input.location!r}: {exc!r}"
                logger.warning(error)
                fetched = None
            if fetched is None and error is None:
                error = f"weather lookup returned nothing for {weather_input.location!r}"
                logger.warning(error)
            if fetched:
                # Fill only gaps; user-provided fields are never overwritten.
                if temperature_c is None and fetched.get("temperature_c") is not None:
                    temperature_c = fetched["temperature_c"]
                    sources["temperature_c"] = "open-meteo"
                if condition is None and fetched.get("condition") is not None:
                    condition = fetched["condition"]
                    sources["condition"] = "open-meteo"
                forecast_note = fetched.get("forecast_note")

    if temperature_c is None and condition is None:
        return None, error

    return (
        WeatherContext(
            temperature_c=temperature_c,
            condition=condition,
            sources=sources,
            forecast_note=forecast_note,
        ),
        error,
    )
