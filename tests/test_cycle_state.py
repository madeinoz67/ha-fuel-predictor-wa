"""Unit tests for FuelPricePredictor.cycle_state (live cycle-position diagnostic)."""

from __future__ import annotations

from datetime import date, timedelta

from custom_components.fuel_predictor_wa.predictor import FuelPricePredictor


def _weekly_series(start: date, cycles: int = 3) -> dict[date, float]:
    """A clean WA-like weekly cycle: jump then -2/day fade. Detects hikes weekly."""
    week = [180.0, 178.0, 176.0, 174.0, 172.0, 170.0, 168.0]
    prices: dict[date, float] = {}
    for i, p in enumerate(week * cycles):
        prices[start + timedelta(days=i)] = p
    return prices


def test_cycle_state_empty_when_unfitted() -> None:
    assert FuelPricePredictor().cycle_state(date(2026, 1, 1)) == {}


def test_cycle_state_empty_for_constant_tier() -> None:
    """< MIN_LEVEL_WINDOW rows -> constant tier, no _last_fit_date -> no cycle."""
    pred = FuelPricePredictor()
    pred.fit({date(2026, 1, 1): 170.0, date(2026, 1, 2): 171.0})
    assert pred.cycle_state(date(2026, 1, 2)) == {}


def test_cycle_state_at_last_fit_day() -> None:
    """After fit, cycle_state at the last training day reports fit-time phase."""
    start = date(2026, 1, 1)
    pred = FuelPricePredictor()
    pred.fit(_weekly_series(start, cycles=3))  # 21 days; hikes at 7, 14
    last_fit_day = start + timedelta(days=20)
    # Days since the last hike (index 14) at index 20 = 6.
    cs = pred.cycle_state(last_fit_day)
    assert cs["cycle_len_days"] == 7
    assert cs["days_since_last_hike"] == 6
    assert cs["cycle_pos"] == 6
    assert cs["expected_next_hike_in_days"] == 1  # 7 - 6


def test_cycle_state_advances_and_wraps_with_elapsed_days() -> None:
    start = date(2026, 1, 1)
    pred = FuelPricePredictor()
    pred.fit(_weekly_series(start, cycles=3))
    last_fit_day = start + timedelta(days=20)
    # One day later: days_since_hike = 7 -> wraps to cycle_pos 0 (a hike is due).
    cs = pred.cycle_state(last_fit_day + timedelta(days=1))
    assert cs["cycle_pos"] == 0
    assert cs["days_since_last_hike"] == 7
    assert cs["expected_next_hike_in_days"] == 7


def test_cycle_state_tolerates_old_pickle_missing_attr() -> None:
    """A model pickled before this attribute existed deserializes without it.

    cycle_state must degrade gracefully (assume "just hiked at fit") rather than
    raise AttributeError — a raise here kills the cheapest_day entity on reload.
    This is the v0.2.4 regression: a v0.2.3-trained model.pkl loaded by v0.2.4.
    """
    pred = FuelPricePredictor()
    pred.fit(_weekly_series(date(2026, 1, 1), cycles=3))
    del pred._days_since_hike_at_fit  # simulate the pre-v0.2.4 pickle
    pred._last_fit_date = date(2026, 1, 21)
    pred._L = 7
    pred._fitted = True
    cs = pred.cycle_state(date(2026, 1, 23))  # 2 days elapsed since fit
    assert cs["cycle_len_days"] == 7
    assert cs["days_since_last_hike"] == 2  # getattr default 0 + 2 elapsed
    assert cs["cycle_pos"] == 2
