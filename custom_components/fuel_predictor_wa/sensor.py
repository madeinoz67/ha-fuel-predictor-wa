"""Sensor platform for Fuel Predictor WA."""

from __future__ import annotations

import logging
from typing import Any

from homeassistant.components.sensor import SensorEntity, SensorStateClass
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from .const import DOMAIN, UNIT_CENTS_PER_LITRE
from .coordinator import FuelPredictorDataUpdateCoordinator
from .predictor import ForecastResult

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up Fuel Predictor WA sensors."""
    coordinator: FuelPredictorDataUpdateCoordinator = hass.data[DOMAIN][entry.entry_id]
    async_add_entities(
        [
            CheapestDaySensor(coordinator, entry),
            CheapestStationTodaySensor(coordinator, entry),
            StatusSensor(coordinator, entry),
        ]
    )


class _FuelPredictorEntity(CoordinatorEntity[FuelPredictorDataUpdateCoordinator], SensorEntity):
    """Common base."""

    _attr_has_entity_name = True
    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_key: str = ""

    def __init__(self, coordinator: FuelPredictorDataUpdateCoordinator, entry: ConfigEntry) -> None:
        super().__init__(coordinator)
        self._attr_unique_id = f"{entry.entry_id}_{self._attr_key}"

    @property
    def _data(self) -> dict[str, Any]:
        return self.coordinator.data or {}


class CheapestDaySensor(_FuelPredictorEntity):
    """Cheapest day in the horizon + its predicted c/L."""

    _attr_key = "cheapest_day"
    _attr_name = "Cheapest day"
    _attr_native_unit_of_measurement = UNIT_CENTS_PER_LITRE
    _attr_icon = "mdi:calendar-star"

    @property
    def native_value(self) -> float | None:
        result: ForecastResult | None = self._data.get("forecast")
        if result and result.cheapest_price is not None:
            return round(result.cheapest_price, 1)
        return None

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        result = self._data.get("forecast")
        if not result:
            return {}
        return {
            "cheapest_date": result.cheapest_day.day.isoformat(),
            "cheapest_source": result.cheapest_day.source,
            "horizon": [
                {"date": p.day.isoformat(), "price": p.price_cpl, "source": p.source}
                for p in result.points
            ],
        }


class CheapestStationTodaySensor(_FuelPredictorEntity):
    """Cheapest nearby station today + a short ranked list."""

    _attr_key = "cheapest_station_today"
    _attr_name = "Cheapest station today"
    _attr_native_unit_of_measurement = UNIT_CENTS_PER_LITRE
    _attr_icon = "mdi:gas-station"

    @property
    def native_value(self) -> float | None:
        today: list[dict[str, Any]] = self._data.get("today") or []
        if today and today[0].get("price") is not None:
            return round(today[0]["price"], 1)
        return None

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        today: list[dict[str, Any]] = self._data.get("today") or []
        top = today[0] if today else {}
        return {
            "brand": top.get("brand"),
            "suburb": top.get("location"),
            "address": top.get("address"),
            "stations": [
                {
                    "brand": s.get("brand"),
                    "suburb": s.get("location"),
                    "address": s.get("address"),
                    "price": s.get("price"),
                }
                for s in today
            ],
        }


class StatusSensor(_FuelPredictorEntity):
    """Diagnostic sensor exposing the training lifecycle state."""

    _attr_key = "status"
    _attr_name = "Training status"
    _attr_icon = "mdi:brain"

    @property
    def native_value(self) -> str | None:
        return self.coordinator.status
