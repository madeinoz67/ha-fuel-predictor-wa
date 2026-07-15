"""Unit tests for the FuelPricePredictor (no HA runtime needed).

Covers:
  - Preserved contract: known-verbatim emission, cheapest() picker, unfit
    behavior.
  - New cycle-aware HGBR model: offset calibration (the 19c post-hike bias
    fix), degradation tiers, causality, clamp behavior, train_metrics.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pytest

from custom_components.fuel_predictor_wa.predictor import (
    CLAMP_HI,
    CLAMP_LO,
    MIN_LEVEL_WINDOW,
    ForecastResult,
    FuelPricePredictor,
    build_row_features,
    detect_hikes,
    median_cycle_len,
)

START = date(2026, 7, 13)


# --- Preserved contract -------------------------------------------------------


def test_predict_returns_known_prices_verbatim() -> None:
    predictor = FuelPricePredictor()
    known = {START: 189.9, START + timedelta(days=1): 192.5}
    points = predictor.predict(START, 2, known)
    assert [p.price_cpl for p in points] == [189.9, 192.5]
    assert all(p.source == "known" for p in points)


def test_cheapest_picks_minimum_priced_day() -> None:
    predictor = FuelPricePredictor()
    known = {START: 189.9, START + timedelta(days=1): 175.0}
    points = predictor.predict(START, 2, known)
    result = ForecastResult(points=points, cheapest_day=FuelPricePredictor.cheapest(points))
    assert result.cheapest_day.day == START + timedelta(days=1)
    assert result.cheapest_price == 175.0


def test_cheapest_raises_value_error_on_empty_horizon() -> None:
    """Defensive guard: empty horizon raises ValueError, not a cryptic IndexError."""
    with pytest.raises(ValueError):
        FuelPricePredictor.cheapest([])


def test_unfit_forecast_yields_none_prices() -> None:
    """An unfitted predictor must emit None prices (no sklearn import path)."""
    predictor = FuelPricePredictor()
    points = predictor.predict(START, 7)
    assert all(p.price_cpl is None for p in points)
    assert all(p.source == "forecast" for p in points)


def test_fit_then_forecast_produces_priced_points() -> None:
    """Fitted predictor emits priced forecast points in a realistic band.

    Note: this was updated from the original 170-200 band (which assumed the
    old average-baseline). The new HGBR model can swing wider on a synthetic
    cyclic series; we assert sane non-negative prices instead of a tight band.
    """
    predictor = FuelPricePredictor()
    # A 7-day sawtooth (WA-like): hike to 180, decay to 168 over the week.
    week = [180.0, 178.0, 176.0, 174.0, 172.0, 170.0, 168.0]
    history = {(date(2026, 5, 1) + timedelta(days=i)): week[i % 7] for i in range(70)}
    predictor.fit(history)
    assert predictor._fitted  # noqa: SLF001

    points = predictor.predict(START, 7)
    forecast = [p for p in points if p.source == "forecast"]
    assert forecast, "expected forecast-day points after fitting"
    assert all(p.price_cpl is not None for p in forecast)
    # Sane fuel-price band; never negative; never absurd.
    assert all(100.0 <= p.price_cpl <= 220.0 for p in forecast)


# --- Offset calibration (the 19c-bias acceptance gate) -----------------------


def test_offset_calibration_pins_level_to_known_anchor() -> None:
    """Fit on history whose recent_mean is ~160, then predict with a 20c hike
    as the known anchor. The forecast must pin to ~180 (the known anchor),
    NOT to ~160 (the stale recent mean).

    This is THE acceptance gate for the post-hike bias fix. If you can't make
    it pass, report BLOCKED rather than weakening the assertion.
    """
    predictor = FuelPricePredictor()
    # ~90 days of essentially flat history at 160 (tiny weekday wiggle so the
    # model has *some* signal but the recent mean is unambiguously ~160). No
    # cycle -> weekday_mean tier (the pure-fade model needs >=3 hikes to fade
    # on); the offset-calibration pin must hold for BOTH tiers.
    history = {(date(2026, 4, 1) + timedelta(days=i)): 160.0 + (i % 7) * 0.1 for i in range(90)}
    predictor.fit(history)
    assert predictor._fitted  # noqa: SLF001
    assert predictor.train_metrics is not None
    # Sanity: the training history really is anchored near 160.
    assert abs(predictor._recent_mean - 160.0) < 1.0  # noqa: SLF001

    # A 20c hike just happened: today's known price is 180.
    known = {START: 180.0}
    points = predictor.predict(START, 7, known)

    # Today is verbatim known.
    assert points[0].day == START
    assert points[0].price_cpl == 180.0
    assert points[0].source == "known"

    # Day-3 forecast must be anchored near the known price (within 3c),
    # NOT near the stale recent mean.
    day3 = points[3]
    assert day3.source == "forecast"
    assert abs(day3.price_cpl - 180.0) < 3.0, (
        f"offset calibration failed: day-3 forecast {day3.price_cpl}c is not "
        f"pinned to the known anchor 180c (the post-hike bias is back)"
    )
    assert abs(day3.price_cpl - 160.0) > 10.0, (
        f"day-3 forecast {day3.price_cpl}c collapsed to the stale recent mean "
        f"~160c -- offset calibration is not engaged"
    )


def test_empirical_fade_makes_forecast_curve_discriminate_cheapest_day() -> None:
    """Regression test for the cheapest-day signal (the ML6 fix).

    The offset-calibrated GBM produced a nearly-flat forecast curve on real
    WA data (cycle_pos went out-of-distribution at predict time), so the
    argmin was noise and cheapest_day_hit_rate sat at the random floor. The
    empirical-fade-anchored forecast must (a) NOT be flat across forecast
    days and (b) put the trough (cheapest day) at max cycle_pos -- the day
    deepest into the fade from a peak anchor.
    """
    predictor = FuelPricePredictor()
    # Clear 7-day cycle: peak 180 at cp=0, monotone fade to trough 168 at cp=6.
    week = [180.0, 178.0, 176.0, 174.0, 172.0, 170.0, 168.0]
    base = date(2026, 4, 1)
    n = 70
    history = {base + timedelta(days=i): week[i % 7] for i in range(n)}
    predictor.fit(history)
    assert predictor._fitted  # noqa: SLF001
    assert predictor.train_metrics is not None
    assert predictor.train_metrics["model_kind"] == "fade"
    # Fade curve must be populated and carry the cycle shape.
    assert predictor._fade_curve, "fade curve not populated for histgbr tier"  # noqa: SLF001
    fade_spread = max(predictor._fade_curve.values()) - min(  # noqa: SLF001
        predictor._fade_curve.values()
    )
    assert fade_spread > 5.0, f"fade curve is flat (spread {fade_spread:.2f})"

    # Training ended at i=69 = cp=6 (trough). The next day (i=70) is cp=0 (peak).
    anchor_day = base + timedelta(days=n)
    known = {anchor_day: 180.0}
    pts = predictor.predict(start=anchor_day, horizon=7, known=known)

    forecast = [p for p in pts if p.source == "forecast"]
    assert len(forecast) >= 5
    prices = [p.price_cpl for p in forecast]
    spread = max(prices) - min(prices)
    assert spread > 3.0, (
        f"forecast is flat (spread {spread:.2f} c/L) -- cheapest-day signal lost; prices={prices}"
    )

    # From a peak anchor the fade is monotone decreasing to the trough, so the
    # cheapest forecast day is the LAST one (max cycle_pos in the window).
    cheapest = min(forecast, key=lambda p: p.price_cpl)
    assert cheapest.day == forecast[-1].day, (
        f"cheapest forecast day {cheapest.day} is not the trough "
        f"(expected {forecast[-1].day}); fade shape is wrong; prices={prices}"
    )
    # And the trough price is near the historical 168 c/L trough.
    assert abs(cheapest.price_cpl - 168.0) < 4.0, (
        f"trough forecast {cheapest.price_cpl} not near historical 168; prices={prices}"
    )


def test_predict_emits_known_days_verbatim() -> None:
    """Multiple known days (within horizon) must be returned exactly."""
    predictor = FuelPricePredictor()
    history = {(date(2026, 5, 1) + timedelta(days=i)): 170.0 for i in range(50)}
    predictor.fit(history)
    known = {
        START: 195.0,
        START + timedelta(days=2): 188.0,
        START + timedelta(days=5): 175.0,
    }
    points = predictor.predict(START, 7, known)
    by_day = {p.day: p for p in points}
    for d, expected in known.items():
        assert by_day[d].price_cpl == expected
        assert by_day[d].source == "known"


# --- Degradation tiers -------------------------------------------------------


def test_degradation_constant_under_7_days() -> None:
    """n < 7 -> constant tier; forecast repeats the latest training price."""
    predictor = FuelPricePredictor()
    history = {
        date(2026, 7, 1): 175.0,
        date(2026, 7, 2): 176.0,
        date(2026, 7, 3): 177.0,
        date(2026, 7, 4): 178.0,
    }
    predictor.fit(history)
    assert predictor.train_metrics is not None
    assert predictor.train_metrics["model_kind"] == "constant"

    points = predictor.predict(START, 5)
    forecast = [p for p in points if p.source == "forecast"]
    assert forecast, "constant tier must still emit forecast points"
    # Latest training price is 178.0 -> every forecast day equals it.
    assert all(p.price_cpl == 178.0 for p in forecast)


def test_degradation_weekday_mean_under_14_days() -> None:
    """7 <= n < 14 -> weekday_mean tier; forecasts are non-negative + finite."""
    predictor = FuelPricePredictor()
    history = {date(2026, 7, 1) + timedelta(days=i): 175.0 + (i % 7) for i in range(10)}
    predictor.fit(history)
    assert predictor.train_metrics is not None
    assert predictor.train_metrics["model_kind"] == "weekday_mean"

    points = predictor.predict(START, 7)
    forecast = [p for p in points if p.source == "forecast"]
    assert forecast
    assert all(p.price_cpl is not None for p in forecast)
    assert all(p.price_cpl >= 0 for p in forecast)


def test_degradation_weekday_mean_when_too_few_hikes() -> None:
    """>=14 days but <3 hikes -> weekday_mean tier (no cycle signal to fade on).

    The old ridge_degraded (sklearn Ridge) tier is gone — sklearn can't install
    on HA's Python 3.14. Without a cycle signal (>=3 hikes), the pure-fade model
    has nothing to fade on, so it falls back to the numpy weekday_mean tier.
    """
    predictor = FuelPricePredictor()
    # 20 days, no clear cycle (so <3 hikes).
    history = {date(2026, 6, 1) + timedelta(days=i): 170.0 + (i % 3) * 0.3 for i in range(20)}
    predictor.fit(history)
    assert predictor.train_metrics is not None
    assert predictor.train_metrics["model_kind"] == "weekday_mean"

    points = predictor.predict(START, 7)
    forecast = [p for p in points if p.source == "forecast"]
    assert forecast
    assert all(p.price_cpl is not None for p in forecast)
    assert all(p.price_cpl >= 0 for p in forecast)


# --- Causality + safety -------------------------------------------------------


def test_features_causal_no_future_leakage() -> None:
    """Features at row t must depend only on prices[0..t-1].

    Inspect the feature builder directly: perturbing prices[t:] must not change
    features at t.
    """
    prices = [160.0 + 0.1 * i for i in range(40)]
    hike_days = detect_hikes(prices)
    cycle_len = 7

    t = 30
    feats_before = build_row_features(prices, t, hike_days, cycle_len)

    # Hammer every price at index >= t with a huge perturbation.
    perturbed = list(prices)
    for i in range(t, len(perturbed)):
        perturbed[i] = perturbed[i] + 1000.0
    feats_after = build_row_features(perturbed, t, hike_days, cycle_len)

    assert feats_before == feats_after, "feature leak: row t changed when only prices[t:] changed"


def test_predict_never_returns_negative() -> None:
    """The clamp must keep every forecast >= 0 even on edge-case inputs."""
    predictor = FuelPricePredictor()
    # A real cyclic history with a low trough.
    week = [180.0, 178.0, 176.0, 174.0, 172.0, 170.0, 168.0]
    history = {(date(2026, 4, 1) + timedelta(days=i)): week[i % 7] for i in range(60)}
    predictor.fit(history)
    points = predictor.predict(START, 14)
    for p in points:
        assert p.price_cpl is None or p.price_cpl >= 0.0


# --- train_metrics ------------------------------------------------------------


def test_train_metrics_populated() -> None:
    """After fit on >=35 days with hikes, train_metrics is fully populated
    and model_kind == fade (the pure-fade production model)."""
    predictor = FuelPricePredictor()
    # 70 days of a clean weekly cycle -> >=3 hikes, n >= 35.
    week = [180.0, 178.0, 176.0, 174.0, 172.0, 170.0, 168.0]
    history = {(date(2026, 4, 1) + timedelta(days=i)): week[i % 7] for i in range(70)}
    predictor.fit(history)

    m = predictor.train_metrics
    assert m is not None
    assert m["model_kind"] == "fade"
    for key in (
        "mae",
        "baseline_mae",
        "improvement_pct",
        "n_train",
        "n_holdout",
        "cycle_len_days",
        "n_hikes",
        "beats_baseline",
    ):
        assert key in m, f"train_metrics missing {key}"
    assert m["n_train"] == 70
    assert m["n_hikes"] >= 3
    assert isinstance(m["beats_baseline"], bool)


# --- walk-forward MAPE internal consistency ----------------------------------


def test_walkforward_mape_pairs_actual_with_own_error() -> None:
    """The walk-forward MAPE must pair each error with its OWN actual.

    Regression: the MAPE list-comprehension once zipped a weakly-guarded
    actuals list against the strictly-guarded errors list (full guard chain:
    prefix long enough, >=3 hikes on the prefix, fade curve non-empty). When a
    hold day was skipped, the two lists had different lengths AND different
    day-identities, so ``zip(..., strict=False)`` silently paired
    ``error[i]`` with ``actual[j]`` for different days. The result was a
    silently-wrong ``mape_pct``.

    This test forces a skip (early hold days whose prefix has <3 hikes) and
    checks internal consistency: ``mape_pct`` == ``mean(err_i / actual_i)``
    over the PREDICTED hold days only. The pure-fade forecast is fully
    deterministic (no stub regressor needed), so the oracle is exact.
    """
    from custom_components.fuel_predictor_wa.predictor import cycle_pos_at

    predictor = FuelPricePredictor()
    # n = 26, 7-day descending sawtooth -> +12 hike at every cycle boundary.
    # Full-series hikes land at indices 7, 14, 21.
    # hold = min(28, 26//5) = 5, step = 1, hold_indices = [21, 22, 23, 24, 25].
    # For each h, prefix = series[:h-2]; need >=3 hikes IN THE PREFIX:
    #   h=21: prefix=0..18 -> hikes {7,14}     = 2 -> SKIPPED
    #   h=22: prefix=0..19 -> hikes {7,14}     = 2 -> SKIPPED
    #   h=23: prefix=0..20 -> hikes {7,14}     = 2 -> SKIPPED
    #   h=24: prefix=0..21 -> hikes {7,14,21}  = 3 -> predicted
    #   h=25: prefix=0..22 -> hikes {7,14,21}  = 3 -> predicted
    # So three hold days are skipped -> the bug would mispair errors.
    n = 26
    week = [180.0, 178.0, 176.0, 174.0, 172.0, 170.0, 168.0]
    series = [week[i % 7] for i in range(n)]
    base = date(2026, 5, 1)
    dates = [base + timedelta(days=i) for i in range(n)]
    # fit() populates instance state (min28/max28) + selects the fade tier.
    predictor.fit(dict(zip(dates, series, strict=True)))

    metrics = predictor._walk_forward(  # noqa: SLF001
        series=series,
        dates=dates,
        hold=5,
    )

    # The walk-forward ran and at least one day was predicted.
    assert metrics["n_holdout"] >= 1, "expected at least one predicted hold day"
    assert metrics["mape_pct"] is not None

    # Oracle: walk the SAME guards and compute the pure-fade forecast for each
    # PREDICTED hold day, then MAPE = mean(err / max(|actual|, 1e-6)) * 100.
    hold_indices = list(range(n - 5, n))
    lo_clamp = CLAMP_LO * min(series)
    hi_clamp = CLAMP_HI * max(series)
    pairs: list[tuple[float, float]] = []
    for h in hold_indices:
        anchor_t = h - 2
        prefix_end = anchor_t
        if prefix_end < MIN_LEVEL_WINDOW + 2:
            continue
        prefix = series[:prefix_end]
        hikes_p = detect_hikes(prefix)
        if len(hikes_p) < 3:
            continue
        L_p = median_cycle_len(hikes_p)
        by_cp: dict[int, list[float]] = {}
        for i, p in enumerate(prefix):
            by_cp.setdefault(cycle_pos_at(i, hikes_p) % L_p, []).append(p)
        fade_p = {cp: float(np.mean(ps)) for cp, ps in by_cp.items()}
        fade_mean_p = float(np.mean(list(fade_p.values())))
        anchor_price = series[anchor_t]
        anchor_cp = cycle_pos_at(anchor_t, hikes_p) % L_p
        target_cp = cycle_pos_at(h, hikes_p) % L_p
        pred = anchor_price + (
            fade_p.get(target_cp, fade_mean_p) - fade_p.get(anchor_cp, fade_mean_p)
        )
        pred = max(lo_clamp, min(hi_clamp, pred))
        actual = series[h]
        pairs.append((actual, abs(actual - pred)))

    # Sanity: the skip really happened (fewer pairs than hold days).
    assert len(pairs) < len(hold_indices), "test setup did not force a skip"
    expected_mape = sum(err / max(abs(a), 1e-6) for a, err in pairs) / len(pairs) * 100.0
    assert metrics["mape_pct"] == pytest.approx(expected_mape, rel=1e-9)
