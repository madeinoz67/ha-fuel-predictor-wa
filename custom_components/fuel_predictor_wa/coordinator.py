"""DataUpdateCoordinator: poll FuelWatch + run the predictor."""

from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path

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
    MODEL_FILENAME,
    PRODUCT_CSV_DESCRIPTION,
    STATUS_ERROR,
    STATUS_READY,
    STATUS_TRAINING,
    STATUS_UNTRAINED,
    STORAGE_DIRNAME,
    UPDATE_INTERVAL_MINUTES,
)
from .fuelwatch import FuelWatchClient
from .historic_client import HistoricClient
from .predictor import ForecastResult, FuelPricePredictor
from .trainer import assemble_and_train, load_model, save_model

_LOGGER = logging.getLogger(__name__)


class FuelPredictorDataUpdateCoordinator(DataUpdateCoordinator):
    """Coordinates live polling, model load/train, and forecasting."""

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        super().__init__(
            hass,
            _LOGGER,
            name=DOMAIN,
            update_interval=timedelta(minutes=UPDATE_INTERVAL_MINUTES),
        )
        self.entry = entry
        # Options override data — changes in the options flow (e.g. horizon,
        # product) must take effect on reload. getattr for robustness.
        cfg = {**entry.data, **getattr(entry, "options", {})}
        self.product: int = cfg[CONF_PRODUCT]
        self.suburb: str = cfg[CONF_SUBURB]
        self.surrounding: bool = cfg.get(CONF_SURROUNDING, DEFAULT_SURROUNDING)
        self.horizon: int = cfg.get(CONF_FORECAST_HORIZON_DAYS, DEFAULT_FORECAST_HORIZON_DAYS)
        self.station_limit: int = cfg.get(CONF_STATION_LIMIT, DEFAULT_STATION_LIMIT)

        self.client = FuelWatchClient(hass)
        self.predictor = FuelPricePredictor()
        self.status: str = STATUS_UNTRAINED
        self._train_in_progress = False
        # Model loading is deferred to async_load_model() so the (blocking) file
        # read runs in the executor, never on the event loop.

    # --- storage + model -------------------------------------------------
    def _storage_dir(self) -> Path:
        uid = self.entry.unique_id or self.entry.entry_id
        return Path(self.hass.config.path(STORAGE_DIRNAME, uid))

    async def async_load_model(self) -> None:
        """Load the trained model artifact (if any), off the event loop."""
        loaded = await self.hass.async_add_executor_job(
            load_model, self._storage_dir() / MODEL_FILENAME
        )
        if loaded is not None:
            self.predictor = loaded
            self.status = STATUS_READY

    # --- training seam (overridden in tests) -----------------------------
    def _build_fetch_month(self):
        """Return an async fetch_month(year, month) bound to the live blob client."""
        historic = HistoricClient(self.hass)
        description = PRODUCT_CSV_DESCRIPTION[self.product]

        async def _fetch(year: int, month: int):
            return await historic.async_fetch_month(year, month, description)

        return _fetch

    async def _async_train_background(self, retrain: bool = False) -> None:
        if self._train_in_progress:
            return
        self._train_in_progress = True
        self.status = STATUS_TRAINING
        self.async_update_listeners()
        try:
            predictor = await assemble_and_train(
                self._build_fetch_month(),
                date.today(),
                executor=self.hass.async_add_executor_job,
            )
            await self.hass.async_add_executor_job(save_model, predictor, self._storage_dir())
            self.predictor = predictor
            self.status = STATUS_READY
            await self.async_request_refresh()
        except Exception as err:  # noqa: BLE001
            _LOGGER.warning("background training failed: %s", err)
            self.status = STATUS_ERROR
            self.async_update_listeners()
        finally:
            self._train_in_progress = False

    async def async_request_retrain(self) -> None:
        await self._async_train_background(retrain=True)

    # --- refresh ---------------------------------------------------------
    async def _async_update_data(self) -> dict:
        try:
            today = await self.client.async_fetch_today(self.product, self.suburb, self.surrounding)
            try:
                tomorrow = await self.client.async_fetch_tomorrow(
                    self.product, self.suburb, self.surrounding
                )
            except Exception:  # noqa: BLE001
                tomorrow = []
        except Exception as err:  # noqa: BLE001
            raise UpdateFailed(f"Error communicating with FuelWatch: {err}") from err

        today_date = date.today()
        known: dict[date, float] = {}
        today_prices = [s["price"] for s in today if s.get("price") is not None]
        if today_prices:
            known[today_date] = min(today_prices)
        if tomorrow:
            tmr = [s["price"] for s in tomorrow if s.get("price") is not None]
            if tmr:
                known[today_date + timedelta(days=1)] = min(tmr)

        # Launch background training if we have no usable model yet.
        if not self.predictor._fitted and not self._train_in_progress:  # noqa: SLF001
            self.hass.async_create_task(self._async_train_background())

        # predict is CPU-bound (pure numpy) -> run it off the event loop.
        points = await self.hass.async_add_executor_job(
            self.predictor.predict, today_date, self.horizon, known
        )
        forecast = ForecastResult(points=points, cheapest_day=FuelPricePredictor.cheapest(points))
        stations_sorted = sorted(today, key=lambda s: s.get("price", float("inf")))[
            : self.station_limit
        ]
        return {"forecast": forecast, "today": stations_sorted}
