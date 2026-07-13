"""Tests for the ModelFitSensor (holdout goodness-of-fit diagnostic)."""

from custom_components.fuel_predictor_wa.sensor import ModelFitSensor


class _Predictor:
    """Minimal stand-in for FuelPricePredictor exposing train_metrics."""

    def __init__(self, metrics):
        self.train_metrics = metrics


class _Coord:
    """Minimal coordinator stand-in: only .predictor is read by the sensor."""

    def __init__(self, metrics):
        self.predictor = _Predictor(metrics)


_FULL = {
    "mae": 3.4,
    "mape_pct": 1.2,
    "baseline_mae": 5.0,
    "improvement_pct": 32.0,
    "post_hike_mae": 4.1,
    "normal_mae": 3.0,
    "model_kind": "histgbr",
    "cycle_len_days": 7,
    "n_hikes": 12,
    "beats_baseline": True,
    "trained_at": "2026-07-13T12:00:00Z",
}


def _build(metrics) -> ModelFitSensor:
    sensor = ModelFitSensor.__new__(ModelFitSensor)
    sensor.coordinator = _Coord(metrics)
    return sensor


def test_native_value_rounds_mae_to_two_decimals() -> None:
    sensor = _build(_FULL)
    assert sensor.native_value == 3.4


def test_native_value_is_none_when_train_metrics_missing() -> None:
    sensor = _build(None)
    assert sensor.native_value is None


def test_native_value_is_none_when_train_metrics_empty() -> None:
    sensor = _build({})
    assert sensor.native_value is None


def test_attributes_contain_expected_keys() -> None:
    attrs = _build(_FULL).extra_state_attributes
    for key in (
        "beats_baseline",
        "model_kind",
        "post_hike_mae",
        "baseline_mae",
        "improvement_pct",
        "mape_pct",
        "cycle_len_days",
        "n_hikes",
        "trained_at",
    ):
        assert key in attrs, f"missing {key}"
    # mae is the native value, so it must NOT be duplicated in attrs.
    assert "mae" not in attrs


def test_attributes_exclude_none_valued_keys() -> None:
    metrics = {
        "mae": None,
        "beats_baseline": None,
        "model_kind": "weekday_mean",
        "post_hike_mae": None,
        "baseline_mae": None,
        "improvement_pct": None,
        "mape_pct": None,
        "cycle_len_days": 7,
        "n_hikes": 0,
        "trained_at": None,
    }
    attrs = _build(metrics).extra_state_attributes
    assert "beats_baseline" not in attrs
    assert "post_hike_mae" not in attrs
    assert attrs["model_kind"] == "weekday_mean"
    assert attrs["cycle_len_days"] == 7


def test_attributes_empty_when_train_metrics_missing() -> None:
    sensor = _build(None)
    assert sensor.extra_state_attributes == {}


def test_attributes_read_defensively_without_predictor_attr() -> None:
    """Sensor must not crash if predictor lacks train_metrics entirely."""

    class _Bare:
        pass

    sensor = ModelFitSensor.__new__(ModelFitSensor)

    class _CoordBare:
        predictor = _Bare()

    sensor.coordinator = _CoordBare()
    assert sensor.native_value is None
    assert sensor.extra_state_attributes == {}
