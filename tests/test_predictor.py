"""Unit tests for the baseline forecaster (no HA runtime needed)."""

from __future__ import annotations

from datetime import date, timedelta

from custom_components.fuel_predictor_wa.predictor import (
    ForecastResult,
    FuelPricePredictor,
)

START = date(2026, 7, 13)


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


def test_fit_then_forecast_produces_priced_points() -> None:
    predictor = FuelPricePredictor()
    history = {(date(2026, 6, 1) + timedelta(days=i)): 180.0 + (i % 7) for i in range(40)}
    predictor.fit(history)
    assert predictor._fitted  # noqa: SLF001

    points = predictor.predict(START, 7)
    forecast = [p for p in points if p.source == "forecast"]
    assert forecast, "expected forecast-day points after fitting"
    assert all(p.price_cpl is not None for p in forecast)
    assert all(170.0 <= p.price_cpl <= 200.0 for p in forecast)


def test_unfit_forecast_yields_none_prices() -> None:
    predictor = FuelPricePredictor()
    points = predictor.predict(START, 7)
    assert all(p.price_cpl is None for p in points)
    assert all(p.source == "forecast" for p in points)
