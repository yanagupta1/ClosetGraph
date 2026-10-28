"""Weather resolution: turn an optional WeatherInput into a resolved
WeatherContext (or nothing), with clear provenance and graceful failure.

The actual Open-Meteo network fetch lands in a later part; it is injected here
as a `fetcher` callable so this logic stays pure and testable without any
external API call.
"""
import logging

from closetgraph.models import WeatherContext, WeatherInput

logger = logging.getLogger(__name__)

# A fetcher takes a location string and returns {"temperature_c": float|None,
# "condition": str|None}, or raises / returns None if it cannot resolve.
Fetcher = "Callable[[str], dict | None]"


def resolve_weather(weather_input: WeatherInput | None, fetcher=None):
    """
    Resolve weather for a request, returning (context, error):

      - context: a WeatherContext when weather was resolved, else None.
      - error:   a short string when a lookup was attempted and failed, else None.

    Rules:
      - No input, or an input with nothing usable -> (None, None). Not an error;
        the request simply carries no weather (Stage 2 behaviour).
      - Explicit user temperature/condition takes precedence over any lookup and
        is tagged source="user". The fetcher is not called in that case.
      - Location-only + a fetcher -> attempt the lookup, tagged
        source="open-meteo". On failure we DO NOT fabricate weather: return
        (None, "<reason>") and log it, so the failure is observable.
    """
    if weather_input is None:
        return None, None

    # User-provided explicit weather wins over any lookup.
    if weather_input.temperature_c is not None or weather_input.condition is not None:
        return (
            WeatherContext(
                temperature_c=weather_input.temperature_c,
                condition=weather_input.condition,
                source="user",
            ),
            None,
        )

    # Only a location given: look it up if we have a fetcher.
    if weather_input.location and fetcher is not None:
        try:
            fetched = fetcher(weather_input.location)
        except Exception as error:  # never fabricate weather on failure
            message = f"weather lookup failed for {weather_input.location!r}: {error!r}"
            logger.warning(message)
            return None, message
        if not fetched:
            message = f"weather lookup returned nothing for {weather_input.location!r}"
            logger.warning(message)
            return None, message
        return (
            WeatherContext(
                temperature_c=fetched.get("temperature_c"),
                condition=fetched.get("condition"),
                source="open-meteo",
            ),
            None,
        )

    # Location given but no fetcher available (e.g. lookup not wired yet).
    if weather_input.location:
        message = f"no weather fetcher available to resolve {weather_input.location!r}"
        logger.warning(message)
        return None, message

    return None, None
