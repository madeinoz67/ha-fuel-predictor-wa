"""Sensor platform for Fuel Predictor WA."""

from __future__ import annotations

import calendar
import logging
from typing import Any

from homeassistant.components.sensor import SensorEntity, SensorStateClass
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from .const import DOMAIN, UNIT_CENTS_PER_LITRE
from .coordinator import FuelPredictorDataUpdateCoordinator
from .predictor import DayForecast, ForecastResult

_LOGGER = logging.getLogger(__name__)

MAX_FORECAST_ENTITIES = 14


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up Fuel Predictor WA sensors."""
    coordinator: FuelPredictorDataUpdateCoordinator = hass.data[DOMAIN][entry.entry_id]
    entities: list[SensorEntity] = [
        CheapestDaySensor(coordinator, entry),
        CheapestStationTodaySensor(coordinator, entry),
        StatusSensor(coordinator, entry),
        ModelFitSensor(coordinator, entry),
        ForecastAccuracySensor(coordinator, entry),
    ]
    for day in range(1, MAX_FORECAST_ENTITIES + 1):
        entities.append(ForecastDaySensor(coordinator, entry, day))
    async_add_entities(entities)


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
        attrs: dict[str, Any] = {
            "cheapest_date": result.cheapest_day.day.isoformat(),
            "cheapest_source": result.cheapest_day.source,
            "horizon": [
                {
                    "date": p.day.isoformat(),
                    "ts": calendar.timegm(p.day.timetuple()) * 1000,
                    "price": p.price_cpl,
                    "source": p.source,
                }
                for p in result.points
            ],
        }
        # Live cycle position: where are we in the price-hike cycle today.
        predictor = getattr(self.coordinator, "predictor", None)
        if predictor is not None and result.points:
            cycle = predictor.cycle_state(result.points[0].day)
            if cycle:
                attrs.update(cycle)
        return attrs


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
    _attr_state_class = None

    @property
    def native_value(self) -> str | None:
        return self.coordinator.status


class ModelFitSensor(_FuelPredictorEntity):
    """Holdout goodness-of-fit metrics for the fitted predictor."""

    _attr_key = "model_fit"
    _attr_name = "Model fit"
    _attr_native_unit_of_measurement = UNIT_CENTS_PER_LITRE
    _attr_icon = "mdi:chart-line"

    @property
    def native_value(self) -> float | None:
        metrics = self._metrics
        mae = metrics.get("mae")
        if mae is None:
            return None
        return round(float(mae), 2)

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        metrics = self._metrics
        return {k: v for k, v in metrics.items() if k != "mae" and v is not None}

    @property
    def _metrics(self) -> dict[str, Any]:
        predictor = getattr(self.coordinator, "predictor", None)
        return getattr(predictor, "train_metrics", None) or {}


class ForecastAccuracySensor(_FuelPredictorEntity):
    """Forecast-vs-actual accuracy from the persisted forecast ledger.

    State = overall MAE (c/L) across all paired forecast/actual days. Attributes
    carry MAE-by-days-out (how accuracy degrades with horizon), recent pairs
    for a scatter view, and coverage so a fresh install is honestly sparse.
    """

    _attr_key = "forecast_accuracy"
    _attr_name = "Forecast accuracy"
    _attr_native_unit_of_measurement = UNIT_CENTS_PER_LITRE
    _attr_icon = "mdi:target"

    @property
    def native_value(self) -> float | None:
        accuracy = self._data.get("accuracy") or {}
        mae = accuracy.get("overall_mae")
        return round(float(mae), 2) if mae is not None else None

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        accuracy = self._data.get("accuracy") or {}
        return {
            "n_pairs": accuracy.get("n_pairs", 0),
            "bias": accuracy.get("bias"),
            "mae_by_days_out": accuracy.get("mae_by_days_out", []),
            "recent": accuracy.get("recent", []),
            "coverage_forecast_days": accuracy.get("coverage_forecast_days", 0),
            "coverage_actual_days": accuracy.get("coverage_actual_days", 0),
        }


class ForecastDaySensor(_FuelPredictorEntity):
    """One day of the forecast horizon as a discrete entity.

    Enables reliable entity-based graphing (apexcharts/mini-graph/history-graph)
    without data_generator — other cards can plot these directly.
    """

    _attr_native_unit_of_measurement = UNIT_CENTS_PER_LITRE
    _attr_icon = "mdi:gas-station"

    def __init__(
        self,
        coordinator: FuelPredictorDataUpdateCoordinator,
        entry: ConfigEntry,
        day: int,
    ) -> None:
        super().__init__(coordinator, entry)
        self._day = day
        self._attr_unique_id = f"{entry.entry_id}_day_{day}"
        self._attr_name = f"Day {day}"

    def _point(self) -> DayForecast | None:
        result = self._data.get("forecast")
        if not result or self._day > len(result.points):
            return None
        return result.points[self._day - 1]

    @property
    def native_value(self) -> float | None:
        p = self._point()
        if p and p.price_cpl is not None:
            return round(p.price_cpl, 1)
        return None

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        p = self._point()
        if not p:
            return {}
        return {"date": p.day.isoformat(), "source": p.source}
