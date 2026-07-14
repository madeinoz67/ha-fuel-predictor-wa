"""Unit tests for the forecast accuracy ledger (no HA runtime needed)."""

from __future__ import annotations

from datetime import date, timedelta

from custom_components.fuel_predictor_wa.forecast_ledger import (
    ACTUALS_LEDGER_FILENAME,
    FORECAST_LEDGER_FILENAME,
    append_actual,
    append_forecast,
    load_accuracy,
)
from custom_components.fuel_predictor_wa.predictor import DayForecast


def _pts(offsets: dict[int, float], start: date, source: str = "forecast") -> list[DayForecast]:
    return [DayForecast(start + timedelta(days=o), p, source) for o, p in offsets.items()]


def test_no_pairs_but_coverage_when_forecast_only(tmp_path) -> None:
    issued = date(2026, 7, 14)
    append_forecast(tmp_path, issued, _pts({0: 176.3, 1: 173.7, 5: 171.0}, issued))
    acc = load_accuracy(tmp_path)
    assert acc["n_pairs"] == 0
    assert acc["overall_mae"] is None
    assert acc["coverage_forecast_days"] == 1
    assert acc["coverage_actual_days"] == 0


def test_append_forecast_idempotent_per_issued_day(tmp_path) -> None:
    issued = date(2026, 7, 14)
    pts = _pts({0: 176.3, 5: 171.0}, issued)
    append_forecast(tmp_path, issued, pts)
    append_forecast(tmp_path, issued, pts)  # second poll the same day
    rows = (tmp_path / FORECAST_LEDGER_FILENAME).read_text().strip().splitlines()
    assert len(rows) == 3  # header + 2 data rows (not doubled)


def test_append_forecast_skips_null_prices(tmp_path) -> None:
    issued = date(2026, 7, 14)
    pts = [
        DayForecast(issued, None, "forecast"),
        DayForecast(issued + timedelta(days=1), 173.7, "forecast"),
    ]
    append_forecast(tmp_path, issued, pts)
    rows = (tmp_path / FORECAST_LEDGER_FILENAME).read_text().strip().splitlines()
    assert len(rows) == 2  # header + 1 (the unfitted null skipped)


def test_append_actual_upserts_same_date(tmp_path) -> None:
    append_actual(tmp_path, date(2026, 7, 14), 176.3)
    append_actual(tmp_path, date(2026, 7, 14), 176.9)  # re-poll overwrites
    rows = (tmp_path / ACTUALS_LEDGER_FILENAME).read_text().strip().splitlines()
    assert len(rows) == 2  # header + 1
    assert "176.9" in rows[1]


def test_append_actual_none_is_noop(tmp_path) -> None:
    append_actual(tmp_path, date(2026, 7, 14), None)
    assert not (tmp_path / ACTUALS_LEDGER_FILENAME).exists()


def test_load_accuracy_pairs_and_mae(tmp_path) -> None:
    issued = date(2026, 7, 10)
    append_forecast(tmp_path, issued, _pts({2: 172.0, 4: 171.0}, issued))
    append_actual(tmp_path, date(2026, 7, 12), 174.0)  # err |172-174|=2.0
    append_actual(tmp_path, date(2026, 7, 14), 170.0)  # err |171-170|=1.0
    acc = load_accuracy(tmp_path)
    assert acc["n_pairs"] == 2
    assert acc["overall_mae"] == 1.5
    by_out = {d["days_out"]: d for d in acc["mae_by_days_out"]}
    assert by_out[2]["mae"] == 2.0
    assert by_out[4]["mae"] == 1.0
    # bias = mean(predicted - actual) = mean(-2.0, 1.0) = -0.5
    assert acc["bias"] == -0.5


def test_load_accuracy_groups_multiple_issuances_per_target(tmp_path) -> None:
    # Target 7/14 forecast from two issuance days at different days-out.
    append_forecast(tmp_path, date(2026, 7, 12), _pts({2: 170.0}, date(2026, 7, 12)))
    append_forecast(tmp_path, date(2026, 7, 13), _pts({1: 171.0}, date(2026, 7, 13)))
    append_actual(tmp_path, date(2026, 7, 14), 172.0)
    acc = load_accuracy(tmp_path)
    assert acc["n_pairs"] == 2
    by_out = {d["days_out"]: d for d in acc["mae_by_days_out"]}
    assert by_out[1]["mae"] == 1.0  # |171-172|
    assert by_out[2]["mae"] == 2.0  # |170-172|


def test_load_accuracy_excludes_old_rows_via_cutoff(tmp_path) -> None:
    old = date.today() - timedelta(days=200)
    append_forecast(tmp_path, old, _pts({1: 171.0}, old))
    append_actual(tmp_path, old + timedelta(days=1), 180.0)
    acc = load_accuracy(tmp_path)  # default max_age_days=180
    assert acc["n_pairs"] == 0


def test_load_accuracy_stub_when_no_ledger(tmp_path) -> None:
    assert load_accuracy(tmp_path) == {
        "overall_mae": None,
        "n_pairs": 0,
        "mae_by_days_out": [],
        "recent": [],
        "coverage_forecast_days": 0,
        "coverage_actual_days": 0,
        "bias": None,
    }
