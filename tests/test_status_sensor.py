"""Tests for the diagnostic status sensor."""

from custom_components.fuel_predictor_wa.const import STATUS_READY
from custom_components.fuel_predictor_wa.sensor import StatusSensor


class _Coord:
    status = STATUS_READY
    data = {}


def test_status_sensor_reads_coordinator_status() -> None:
    sensor = StatusSensor.__new__(StatusSensor)
    sensor.coordinator = _Coord()
    sensor._attr_key = "status"
    assert sensor.native_value == STATUS_READY
