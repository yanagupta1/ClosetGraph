"""Deterministic, stdlib-only parsing of a natural date/time hint into a target
date and an optional time-of-day window.

No LLM and no third-party dependency, so this stays at zero cost and is fully
unit-testable with an injectable reference date. The scope is intentionally
narrow: common relative phrases (today, tonight, tomorrow, this/next <weekday>,
in N days), ISO dates, and time-of-day / meal hints (morning, evening, dinner,
7pm). Anything it does not recognize returns None, so the caller treats the
request as date-agnostic (current conditions) rather than guessing a date.

A time window is only produced when the hint actually names a time, period, or
meal. A pure date hint ("next Friday") stays date-only, which the weather layer
resolves with a daily forecast rather than an hourly one.
"""
from dataclasses import dataclass
from datetime import date, timedelta
import re

_WEEKDAYS = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}

# Period / meal words -> an [start, end) hour window (end exclusive) and a label.
# Kept coarse on purpose; the weather layer averages the hours in the window.
_PERIODS = {
    "morning": (6, 12, "morning"),
    "noon": (11, 14, "midday"),
    "midday": (11, 14, "midday"),
    "afternoon": (12, 17, "afternoon"),
    "evening": (17, 22, "evening"),
    "night": (20, 24, "night"),
}
_MEALS = {
    "breakfast": (7, 10, "breakfast"),
    "brunch": (10, 13, "brunch"),
    "lunch": (12, 14, "lunch"),
    "dinner": (18, 22, "dinner"),
    "supper": (18, 22, "supper"),
}

# "7pm", "7 pm", "7:30pm", and 24-hour "19:00".
_CLOCK_12H_RE = re.compile(r"\b(\d{1,2})(?::(\d{2}))?\s*([ap]m)\b")
_CLOCK_24H_RE = re.compile(r"\b([01]?\d|2[0-3]):([0-5]\d)\b")
_ISO_DATE_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")
_IN_N_DAYS_RE = re.compile(r"\bin\s+(\d{1,2})\s+days?\b")


@dataclass(frozen=True)
class TimeWindow:
    start_hour: int  # inclusive
    end_hour: int  # exclusive
    label: str


@dataclass(frozen=True)
class ParsedWhen:
    target_date: date
    time_window: TimeWindow | None
    note: str  # human-readable summary for provenance, e.g. "2026-11-14 evening"


def _resolve_weekday(text: str, today: date) -> date | None:
    for name, wd in _WEEKDAYS.items():
        if re.search(rf"\b{name}\b", text):
            days = (wd - today.weekday()) % 7
            # "next <weekday>" lands at least a week out; a bare/"this" weekday
            # that equals today resolves to today.
            if "next" in text and days == 0:
                days = 7
            return today + timedelta(days=days)
    return None


def _resolve_date(text: str, today: date) -> date | None:
    iso = _ISO_DATE_RE.search(text)
    if iso:
        try:
            return date.fromisoformat(iso.group(1))
        except ValueError:
            return None
    in_n = _IN_N_DAYS_RE.search(text)
    if in_n:
        return today + timedelta(days=int(in_n.group(1)))
    if "day after tomorrow" in text:
        return today + timedelta(days=2)
    if "tomorrow" in text:
        return today + timedelta(days=1)
    if "tonight" in text or "today" in text:
        return today
    return _resolve_weekday(text, today)


def _resolve_window(text: str) -> TimeWindow | None:
    clock = _CLOCK_12H_RE.search(text)
    if clock:
        hour = int(clock.group(1)) % 12
        if clock.group(3) == "pm":
            hour += 12
        label = clock.group(0).strip()
        return TimeWindow(hour, min(hour + 1, 24), label)
    clock24 = _CLOCK_24H_RE.search(text)
    if clock24:
        hour = int(clock24.group(1))
        return TimeWindow(hour, min(hour + 1, 24), clock24.group(0))
    for word, (start, end, label) in {**_MEALS, **_PERIODS}.items():
        if re.search(rf"\b{word}\b", text):
            return TimeWindow(start, end, label)
    if "tonight" in text:
        start, end, label = _PERIODS["evening"]
        return TimeWindow(start, end, label)
    return None


def parse_when(when: str | None, today: date | None = None) -> ParsedWhen | None:
    """
    Parse a date/time hint into a ParsedWhen, or None when nothing is recognized.

    `today` is injectable for deterministic tests; it defaults to the real
    current date. A window is attached only when the hint names a time, period,
    or meal; a pure date hint stays date-only. When only a time is given
    ("this evening") the date defaults to today.
    """
    if not when or not when.strip():
        return None
    today = today or date.today()
    text = when.lower()

    window = _resolve_window(text)
    target = _resolve_date(text, today)
    if target is None:
        # A bare time-of-day hint ("this evening", "at 7pm") means today.
        if window is None:
            return None
        target = today

    if window is not None:
        note = f"{target.isoformat()} {window.label}"
    else:
        note = target.isoformat()
    return ParsedWhen(target_date=target, time_window=window, note=note)
