"""On-device trainer: fetch N monthly CSVs -> fit -> pickle artifact."""

from __future__ import annotations

import logging
import pickle
from collections.abc import Awaitable, Callable
from datetime import date
from pathlib import Path
from typing import Any

from .const import HISTORY_MONTHS_TARGET, MIN_MONTHS_TO_TRAIN, MODEL_FILENAME
from .historic_client import trailing_months
from .predictor import FuelPricePredictor

_LOGGER = logging.getLogger(__name__)

MODEL_VERSION = 1

# fetch_month(year, month) -> normalized records (already product-filtered).
MonthFetcher = Callable[[int, int], Awaitable[list[dict[str, Any]]]]


def series_from_records(records: list[dict[str, Any]]) -> dict[date, float]:
    """Collapse records to {date: min_price_for_day}."""
    series: dict[date, float] = {}
    for r in records:
        d, p = r["date"], r["price"]
        if d not in series or p < series[d]:
            series[d] = p
    return series


def fit_predictor(series: dict[date, float]) -> FuelPricePredictor:
    predictor = FuelPricePredictor()
    predictor.fit(series)
    return predictor


def save_model(predictor: FuelPricePredictor, storage_dir: Path) -> Path:
    storage_dir.mkdir(parents=True, exist_ok=True)
    artifact = storage_dir / MODEL_FILENAME
    with artifact.open("wb") as fh:
        pickle.dump({"version": MODEL_VERSION, "predictor": predictor}, fh)
    return artifact


def load_model(artifact: Path) -> FuelPricePredictor | None:
    if not artifact.exists():
        return None
    try:
        with artifact.open("rb") as fh:
            data = pickle.load(fh)  # noqa: S301 — our own trusted artifact
    except (OSError, pickle.PickleError, EOFError) as err:
        _LOGGER.warning("Ignoring unreadable model artifact %s: %s", artifact, err)
        return None
    if not isinstance(data, dict) or data.get("version") != MODEL_VERSION:
        _LOGGER.warning(
            "Model artifact version mismatch: %s",
            data.get("version") if isinstance(data, dict) else type(data),
        )
        return None
    return data.get("predictor")


async def assemble_and_train(
    fetch_month: MonthFetcher, today: date, months: int = HISTORY_MONTHS_TARGET
) -> FuelPricePredictor:
    """Fetch `months` months via `fetch_month`, fit, and return the predictor.

    Raises RuntimeError if fewer than MIN_MONTHS_TO_TRAIN months yielded records.
    """
    fetched = 0
    records: list[dict[str, Any]] = []
    for year, month in trailing_months(today, months):
        try:
            month_records = await fetch_month(year, month)
        except Exception as err:  # noqa: BLE001 — one bad month must not abort training
            _LOGGER.warning("historic fetch failed %d-%02d: %s", year, month, err)
            continue
        if month_records:
            fetched += 1
            records.extend(month_records)
    if fetched < MIN_MONTHS_TO_TRAIN:
        raise RuntimeError(f"training needs >= {MIN_MONTHS_TO_TRAIN} months, got {fetched}")
    return fit_predictor(series_from_records(records))
