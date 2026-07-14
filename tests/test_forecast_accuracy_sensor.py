"""Tests for ForecastAccuracySensor + cycle attrs on CheapestDaySensor."""

from __future__ import annotations

from datetime import date, timedelta

from custom_components.fuel_predictor_wa.predictor import (
    DayForecast,
    ForecastResult,
    FuelPricePredictor,
)
from custom_components.fuel_predictor_wa.sensor import (
    CheapestDaySensor,
    ForecastAccuracySensor,
)


# ---------------------------------------------------------------------------
# ForecastAccuracySensor
# ---------------------------------------------------------------------------
class _AccuracyCoord:
    """Minimal coordinator stand-in: only .data is read by the sensor."""

    def __init__(self, data):
        self.data = data


def _acc_sensor(accuracy) -> ForecastAccuracySensor:
    sensor = ForecastAccuracySensor.__new__(ForecastAccuracySensor)
    sensor.coordinator = _AccuracyCoord({"accuracy": accuracy})
    return sensor


_FULL_ACCURACY = {
    "overall_mae": 2.31,
    "n_pairs": 18,
    "bias": -0.4,
    "mae_by_days_out": [
        {"days_out": 1, "mae": 1.2, "n": 9},
        {"days_out": 5, "mae": 3.4, "n": 9},
    ],
    "recent": [{"target_date": "2026-07-13", "predicted": 171.0, "actual": 172.5, "error": 1.5}],
    "coverage_forecast_days": 14,
    "coverage_actual_days": 10,
}


def test_accuracy_native_value_is_overall_mae() -> None:
    assert _acc_sensor(_FULL_ACCURACY).native_value == 2.31


def test_accuracy_native_value_none_when_missing() -> None:
    assert _acc_sensor({}).native_value is None
    assert _acc_sensor(None).native_value is None


def test_accuracy_attributes_carry_bundle() -> None:
    attrs = _acc_sensor(_FULL_ACCURACY).extra_state_attributes
    assert attrs["n_pairs"] == 18
    assert attrs["bias"] == -0.4
    assert attrs["coverage_forecast_days"] == 14
    assert len(attrs["mae_by_days_out"]) == 2
    assert attrs["recent"][0]["error"] == 1.5


def test_accuracy_attributes_safe_when_empty() -> None:
    attrs = _acc_sensor({}).extra_state_attributes
    assert attrs["n_pairs"] == 0
    assert attrs["mae_by_days_out"] == []


# ---------------------------------------------------------------------------
# CheapestDaySensor cycle-state attrs
# ---------------------------------------------------------------------------
class _PredictorStub:
    """Stand-in exposing only cycle_state (what CheapestDaySensor reads)."""

    def __init__(self, cycle):
        self._cycle = cycle

    def cycle_state(self, anchor_date):  # noqa: ARG002
        return self._cycle


def _cheapest_sensor(forecast, predictor) -> CheapestDaySensor:
    sensor = CheapestDaySensor.__new__(CheapestDaySensor)

    class _Coord:
        def __init__(self, forecast, predictor):
            self.data = {"forecast": forecast}
            self.predictor = predictor

    sensor.coordinator = _Coord(forecast, predictor)
    return sensor


def _result(prices: list[float], start: date = date(2026, 7, 14)) -> ForecastResult:
    points = [
        DayForecast(start + timedelta(days=i), p, "forecast") for i, p in enumerate(prices)
    ]
    return ForecastResult(points=points, cheapest_day=FuelPricePredictor.cheapest(points))


def test_cheapest_day_attrs_include_cycle_state() -> None:
    cycle = {
        "cycle_pos": 3,
        "cycle_len_days": 7,
        "days_since_last_hike": 3,
        "expected_next_hike_in_days": 4,
    }
    sensor = _cheapest_sensor(_result([176.3, 173.7, 171.0]), _PredictorStub(cycle))
    attrs = sensor.extra_state_attributes
    assert attrs["cycle_pos"] == 3
    assert attrs["expected_next_hike_in_days"] == 4
    assert "horizon" in attrs  # existing attrs preserved


def test_cheapest_day_attrs_without_cycle_state() -> None:
    """Degraded/unfitted model -> empty cycle -> no cycle keys, horizon intact."""
    sensor = _cheapest_sensor(_result([176.3]), _PredictorStub({}))
    attrs = sensor.extra_state_attributes
    assert "cycle_pos" not in attrs
    assert len(attrs["horizon"]) == 1


def test_cheapest_day_attrs_empty_when_no_forecast() -> None:
    sensor = _cheapest_sensor(None, _PredictorStub({"cycle_pos": 1}))
    assert sensor.extra_state_attributes == {}
