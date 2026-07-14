"""DataUpdateCoordinator: poll FuelWatch + run the predictor."""

from __future__ import annotations

import logging
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator, UpdateFailed

from . import forecast_ledger
from .catchment import (
    async_fetch_suburbs,
    is_cache_fresh,
    load_cached_catchment,
    resolve_catchment,
    save_cached_catchment,
)
from .const import (
    CONF_FORECAST_HORIZON_DAYS,
    CONF_MIN_STATIONS,
    CONF_PRODUCT,
    CONF_STATION_LIMIT,
    CONF_SUBURB,
    CONF_SURROUNDING,
    DEFAULT_FORECAST_HORIZON_DAYS,
    DEFAULT_MIN_STATIONS,
    DEFAULT_STATION_LIMIT,
    DEFAULT_SURROUNDING,
    DOMAIN,
    MODEL_FILENAME,
    PRODUCT_CSV_DESCRIPTION,
    RETRAIN_INTERVAL_HOURS,
    STATUS_ERROR,
    STATUS_READY,
    STATUS_TRAINING,
    STATUS_UNTRAINED,
    STORAGE_DIRNAME,
    UPDATE_INTERVAL_MINUTES,
)
from .fuelwatch import FuelWatchClient
from .historic_client import async_fetch_month_cached
from .predictor import ForecastResult, FuelPricePredictor
from .trainer import assemble_and_train, load_model, save_model

_LOGGER = logging.getLogger(__name__)


def _persist_and_score_ledger(storage_dir, issued_date, points, actual_cpl):
    """Append today's forecast snapshot + actual to the ledger, then score it.

    Runs in the executor (file I/O). Coordinator wraps the call so any failure
    here is logged-and-skipped, never breaking the forecast.
    """
    forecast_ledger.append_forecast(storage_dir, issued_date, points)
    forecast_ledger.append_actual(storage_dir, issued_date, actual_cpl)
    return forecast_ledger.load_accuracy(storage_dir)


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
        self.min_stations: int = cfg.get(CONF_MIN_STATIONS, DEFAULT_MIN_STATIONS)

        self.client = FuelWatchClient(hass)
        self.predictor = FuelPricePredictor()
        self.status: str = STATUS_UNTRAINED
        self._train_in_progress = False
        # Resolved local catchment ( suburb set + metadata ) or None for WA-wide.
        # Lazy-resolved on first train; cached on disk keyed by config.
        self.catchment: dict | None = None
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

    # --- catchment + refit cadence --------------------------------------
    async def _async_resolve_catchment(self) -> dict | None:
        """Resolve + cache the local catchment; None falls back to WA-wide."""
        cached = await self.hass.async_add_executor_job(load_cached_catchment, self._storage_dir())
        if is_cache_fresh(cached, self.product, self.suburb, self.min_stations):
            self.catchment = cached
            return cached
        suburbs = await async_fetch_suburbs(self.hass)
        resolved = (
            resolve_catchment(suburbs or [], self.suburb, self.min_stations) if suburbs else None
        )
        await self.hass.async_add_executor_job(save_cached_catchment, self._storage_dir(), resolved)
        if resolved is not None:
            _LOGGER.info(
                "local catchment for %s: %d suburbs, %d stations",
                resolved["anchor"],
                len(resolved["suburbs"]),
                resolved["total_stations"],
            )
        else:
            _LOGGER.info("catchment unresolved for '%s'; training WA-wide", self.suburb)
        self.catchment = resolved
        return resolved

    def _should_refit(self) -> bool:
        """True if the fitted model is older than the refit interval and idle."""
        if self._train_in_progress:
            return False
        metrics = getattr(self.predictor, "train_metrics", None) or {}
        trained_at = metrics.get("trained_at")
        if not trained_at:
            return False
        try:
            last = datetime.fromisoformat(trained_at)
        except ValueError:
            return False
        if last.tzinfo is None:
            last = last.replace(tzinfo=UTC)
        return datetime.now(UTC) - last >= timedelta(hours=RETRAIN_INTERVAL_HOURS)

    # --- training seam (overridden in tests) -----------------------------
    def _build_fetch_month(self):
        """Cached bulk fetcher: immutable months load from disk; the current +
        previous month re-download (they still gain days). Catchment-agnostic —
        the suburb filter is applied at series collapse."""
        description = PRODUCT_CSV_DESCRIPTION[self.product]
        storage_dir = self._storage_dir()

        async def _fetch(year: int, month: int):
            return await async_fetch_month_cached(self.hass, storage_dir, year, month, description)

        return _fetch

    async def _async_train_background(self, retrain: bool = False) -> None:
        if self._train_in_progress:
            return
        self._train_in_progress = True
        self.status = STATUS_TRAINING
        self.async_update_listeners()
        try:
            await self._async_resolve_catchment()
            suburbs_filter = set(self.catchment["suburbs"]) if self.catchment else None
            predictor = await assemble_and_train(
                self._build_fetch_month(),
                date.today(),
                executor=self.hass.async_add_executor_job,
                suburbs_filter=suburbs_filter,
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

        # Launch background training if we have no usable model yet, else refit
        # on the configured cadence (catchment-aware) after a successful poll.
        if not self.predictor._fitted and not self._train_in_progress:  # noqa: SLF001
            self.hass.async_create_task(self._async_train_background())
        elif self._should_refit():
            self.hass.async_create_task(self._async_train_background(retrain=True))

        # predict is CPU-bound (pure numpy) -> run it off the event loop.
        points = await self.hass.async_add_executor_job(
            self.predictor.predict, today_date, self.horizon, known
        )
        forecast = ForecastResult(points=points, cheapest_day=FuelPricePredictor.cheapest(points))
        stations_sorted = sorted(today, key=lambda s: s.get("price", float("inf")))[
            : self.station_limit
        ]
        # Forecast accuracy ledger: snapshot today's forecast + record today's
        # actual, then score forecast-vs-actual. Best-effort — a ledger failure
        # must never break the forecast. File I/O runs in the executor.
        actual_today = known.get(today_date)
        try:
            accuracy = await self.hass.async_add_executor_job(
                _persist_and_score_ledger,
                self._storage_dir(),
                today_date,
                points,
                actual_today,
            )
        except Exception as err:  # noqa: BLE001
            _LOGGER.debug("forecast ledger update skipped: %s", err)
            accuracy = {}
        return {"forecast": forecast, "today": stations_sorted, "accuracy": accuracy}
