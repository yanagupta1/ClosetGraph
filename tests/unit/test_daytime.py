from datetime import date

from closetgraph.daytime import parse_when

# Fixed reference date so relative phrases resolve deterministically.
# 2026-11-10 is a Tuesday.
TODAY = date(2026, 11, 10)


def test_none_and_blank_return_none():
    assert parse_when(None, TODAY) is None
    assert parse_when("   ", TODAY) is None


def test_unrecognized_hint_returns_none():
    # No date and no time word -> caller treats the request as date-agnostic.
    assert parse_when("sometime soon-ish", TODAY) is None


def test_today_is_date_only():
    parsed = parse_when("today", TODAY)
    assert parsed.target_date == date(2026, 11, 10)
    assert parsed.time_window is None  # date-only stays daily
    assert parsed.note == "2026-11-10"


def test_tomorrow_and_day_after():
    assert parse_when("tomorrow", TODAY).target_date == date(2026, 11, 11)
    assert parse_when("day after tomorrow", TODAY).target_date == date(2026, 11, 12)


def test_in_n_days():
    assert parse_when("in 3 days", TODAY).target_date == date(2026, 11, 13)


def test_bare_weekday_is_upcoming_including_today():
    assert parse_when("friday", TODAY).target_date == date(2026, 11, 13)
    assert parse_when("saturday", TODAY).target_date == date(2026, 11, 14)
    # Today is Tuesday; a bare "tuesday" means today.
    assert parse_when("tuesday", TODAY).target_date == date(2026, 11, 10)


def test_next_weekday_forces_a_week_out_when_it_lands_on_today():
    assert parse_when("next tuesday", TODAY).target_date == date(2026, 11, 17)
    # A non-today weekday is unaffected by "next".
    assert parse_when("next friday", TODAY).target_date == date(2026, 11, 13)


def test_iso_date():
    assert parse_when("2026-12-25", TODAY).target_date == date(2026, 12, 25)


def test_period_attaches_window_and_keeps_date():
    parsed = parse_when("tomorrow evening", TODAY)
    assert parsed.target_date == date(2026, 11, 11)
    assert (parsed.time_window.start_hour, parsed.time_window.end_hour) == (17, 22)
    assert parsed.time_window.label == "evening"
    assert parsed.note == "2026-11-11 evening"


def test_meal_hint_attaches_window():
    parsed = parse_when("dinner on saturday", TODAY)
    assert parsed.target_date == date(2026, 11, 14)
    assert (parsed.time_window.start_hour, parsed.time_window.end_hour) == (18, 22)
    assert parsed.time_window.label == "dinner"


def test_bare_time_of_day_defaults_to_today():
    parsed = parse_when("this evening", TODAY)
    assert parsed.target_date == date(2026, 11, 10)
    assert parsed.time_window.label == "evening"


def test_tonight_is_today_evening():
    parsed = parse_when("tonight", TODAY)
    assert parsed.target_date == date(2026, 11, 10)
    assert parsed.time_window.label == "evening"


def test_clock_time_12h():
    parsed = parse_when("friday at 7pm", TODAY)
    assert parsed.target_date == date(2026, 11, 13)
    assert (parsed.time_window.start_hour, parsed.time_window.end_hour) == (19, 20)


def test_clock_time_24h():
    parsed = parse_when("saturday 19:00", TODAY)
    assert parsed.target_date == date(2026, 11, 14)
    assert parsed.time_window.start_hour == 19
