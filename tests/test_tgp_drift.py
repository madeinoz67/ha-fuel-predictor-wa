"""Unit tests for the wholesale-TGP leading-indicator drift term."""

from __future__ import annotations

from datetime import date, timedelta

from custom_components.fuel_predictor_wa.predictor import (
    TGP_LAG_DAYS,
    FuelPricePredictor,
    _tgp_return,
)


def _weekly_series(start: date, cycles: int) -> dict[date, float]:
    """Clean weekly hike-then-fade cycle; >=3 cycles => fade tier."""
    week = [180.0, 178.0, 176.0, 174.0, 172.0, 170.0, 168.0]
    out: dict[date, float] = {}
    for i, p in enumerate(week * cycles):
        out[start + timedelta(days=i)] = p
    return out


# ---------------------------------------------------------------------------
# _tgp_return (pure helper)
# ---------------------------------------------------------------------------
def test_tgp_return_basic() -> None:
    tgp = [(date(2026, 1, 1), 100.0), (date(2026, 1, 11), 110.0), (date(2026, 1, 21), 121.0)]
    # anchor 2026-01-21, lag 10 => 121 / 110 - 1 = 0.10
    assert abs(_tgp_return(tgp, date(2026, 1, 21), 10) - 0.10) < 1e-9


def test_tgp_return_uses_at_or_before_when_exact_missing() -> None:
    tgp = [(date(2026, 1, 1), 100.0), (date(2026, 1, 15), 120.0)]
    # anchor 2026-01-20 (no exact point; use 01-15=120), lag 10 => past at/before 01-10 = 01-01=100
    assert abs(_tgp_return(tgp, date(2026, 1, 20), 10) - 0.20) < 1e-9


def test_tgp_return_none_when_past_missing() -> None:
    tgp = [(date(2026, 1, 15), 120.0)]  # nothing 10 days before a 01-15 anchor
    assert _tgp_return(tgp, date(2026, 1, 15), 10) is None


def test_tgp_return_none_for_empty_series() -> None:
    assert _tgp_return([], date(2026, 1, 15), 10) is None


# ---------------------------------------------------------------------------
# fit() plumbing: accepts tgp, stores it, exposes tgp_beta in metrics
# ---------------------------------------------------------------------------
def test_fit_accepts_tgp_and_exposes_beta_key() -> None:
    start = date(2026, 1, 1)
    series = _weekly_series(start, cycles=8)  # long enough for a fade-tier holdout
    tgp = {start + timedelta(days=i): 100.0 + i * 0.5 for i in range(120)}
    pred = FuelPricePredictor()
    pred.fit(series, tgp_series=tgp)
    assert pred._tgp_series == tgp  # noqa: SLF001
    assert "tgp_beta" in (pred.train_metrics or {})
    assert "tgp_lag" in (pred.train_metrics or {})


def test_fit_without_tgp_has_no_beta() -> None:
    pred = FuelPricePredictor()
    pred.fit(_weekly_series(date(2026, 1, 1), cycles=8))
    assert pred._tgp_beta is None  # noqa: SLF001
    assert (pred.train_metrics or {}).get("tgp_beta") is None


# ---------------------------------------------------------------------------
# Drift application in predict()
# ---------------------------------------------------------------------------
def test_drift_shifts_forecast_by_beta_times_return() -> None:
    start = date(2026, 1, 1)
    series = _weekly_series(start, cycles=8)
    pred = FuelPricePredictor()
    pred.fit(series)
    assert pred._model_kind == "fade"  # noqa: SLF001

    anchor = max(series)  # last training day
    known = {anchor: series[anchor]}

    # Baseline: no drift.
    pred._tgp_beta = None  # noqa: SLF001
    base = pred.predict(anchor, 5, known=known)

    # With drift: β=20, TGP rose 5% over the lag window => +1.0 c/L level shift.
    pred._tgp_beta = 20.0  # noqa: SLF001
    pred._tgp_lag = TGP_LAG_DAYS  # noqa: SLF001
    pred._tgp_series = {anchor - timedelta(days=TGP_LAG_DAYS): 100.0, anchor: 105.0}  # noqa: SLF001
    drifted = pred.predict(anchor, 5, known=known)

    # Every forecast (non-known) day should be exactly 1.0 c/L higher with drift.
    d_map = {p.day: p.price_cpl for p in drifted}
    for p in base:
        if p.source == "forecast" and p.day in d_map:
            assert abs(d_map[p.day] - p.price_cpl - 1.0) < 0.05, (p.day, d_map[p.day], p.price_cpl)


def test_old_pickle_without_tgp_fields_predicts_without_crash() -> None:
    """A model pickled before the TGP fields existed must still predict."""
    pred = FuelPricePredictor()
    pred.fit(_weekly_series(date(2026, 1, 1), cycles=8))
    # Simulate a pre-v0.4.0 pickle: strip the new fields.
    del pred._tgp_beta  # noqa: SLF001
    del pred._tgp_series  # noqa: SLF001
    del pred._tgp_lag  # noqa: SLF001
    anchor = date(2026, 1, 1) + timedelta(days=55)
    pts = pred.predict(anchor, 5, known={anchor: 170.0})
    assert len(pts) == 5  # no crash, no drift
