"""DataUpdateCoordinator: poll FuelWatch + run the predictor."""
from __future__ import annotations

import logging
from datetime import date, timedelta

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator, UpdateFailed

from .const import (
    CONF_FORECAST_HORIZON_DAYS,
    CONF_PRODUCT,
    CONF_STATION_LIMIT,
    CONF_SUBURB,
    CONF_SURROUNDING,
    DEFAULT_FORECAST_HORIZON_DAYS,
    DEFAULT_STATION_LIMIT,
    DEFAULT_SURROUNDING,
    DOMAIN,
    HISTORY_FILENAME,
    UPDATE_INTERVAL_MINUTES,
)
from .fuelwatch import FuelWatchClient
from .history import load_history, window
from .predictor import ForecastResult, FuelPricePredictor

_LOGGER = logging.getLogger(__name__)

HISTORY_LOOKBACK_DAYS = 365


class FuelPredictorDataUpdateCoordinator(DataUpdateCoordinator):
    """Coordinates live FuelWatch polling, history, and forecasting."""

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        super().__init__(
            hass,
            _LOGGER,
            name=DOMAIN,
            update_interval=timedelta(minutes=UPDATE_INTERVAL_MINUTES),
        )
        self.entry = entry
        data = entry.data
        self.product: int = data[CONF_PRODUCT]
        self.suburb: str = data[CONF_SUBURB]
        self.surrounding: bool = data.get(CONF_SURROUNDING, DEFAULT_SURROUNDING)
        self.horizon: int = data.get(CONF_FORECAST_HORIZON_DAYS, DEFAULT_FORECAST_HORIZON_DAYS)
        self.station_limit: int = data.get(CONF_STATION_LIMIT, DEFAULT_STATION_LIMIT)
        self.client = FuelWatchClient(hass)
        self.predictor = FuelPricePredictor()

    def _fit_from_local_history(self) -> None:
        """Fit the predictor from the local daily-append history, if present."""
        path = self.hass.config.path(HISTORY_FILENAME)
        rows = load_history(path)
        if not rows:
            return
        series = window(rows, HISTORY_LOOKBACK_DAYS)
        self.predictor.fit(series)

    async def _async_update_data(self) -> dict:
        """Fetch live prices + produce forecast + cheapest-stations list."""
        try:
            today = await self.client.async_fetch_today(
                self.product, self.suburb, self.surrounding
            )
            try:
                tomorrow = await self.client.async_fetch_tomorrow(
                    self.product, self.suburb, self.surrounding
                )
            except Exception:  # noqa: BLE001 — tomorrow unavailable before 14:30
                tomorrow = []
        except Exception as err:  # noqa: BLE001
            raise UpdateFailed(f"Error communicating with FuelWatch: {err}") from err

        today_date = date.today()
        known: dict[date, float] = {}
        today_prices = [s["price"] for s in today if s.get("price") is not None]
        if today_prices:
            known[today_date] = min(today_prices)
        if tomorrow:
            tmr_prices = [s["price"] for s in tomorrow if s.get("price") is not None]
            if tmr_prices:
                known[today_date + timedelta(days=1)] = min(tmr_prices)

        self._fit_from_local_history()
        points = self.predictor.predict(today_date, self.horizon, known)
        forecast = ForecastResult(points=points, cheapest_day=FuelPricePredictor.cheapest(points))

        stations_sorted = sorted(
            today, key=lambda s: s.get("price", float("inf"))
        )[: self.station_limit]

        return {"forecast": forecast, "today": stations_sorted}
