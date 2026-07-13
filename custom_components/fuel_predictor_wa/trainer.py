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

MODEL_VERSION = 2

# fetch_month(year, month) -> normalized records (already product-filtered).
MonthFetcher = Callable[[int, int], Awaitable[list[dict[str, Any]]]]

# executor(fn, *args) -> Awaitable[result]. Mirrors hass.async_add_executor_job
# so the CPU-bound fit can run off the HA event loop; the default just runs
# the fn inline so tests without HA still work.
Executor = Callable[..., Awaitable[Any]]


def _get_sklearn_version() -> str | None:
    """Return the installed sklearn version, or None if sklearn is absent.

    sklearn is imported lazily by predictor.fit(); on an unfitted predictor
    (or a fresh interpreter without sklearn) the import must be guarded.
    """
    try:
        import sklearn  # noqa: PLC0415 — lazy on purpose
    except ImportError:
        return None
    return getattr(sklearn, "__version__", None)


async def _default_executor(fn: Callable[..., Any], *args: Any) -> Any:
    """Inline executor: runs fn(*args) off the caller's awaitable context.

    Used when no executor is passed to assemble_and_train so the call stays
    testable without HA while still keeping fit_predictor off the
    implementer's direct return path.
    """
    return fn(*args)


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
    payload = {
        "version": MODEL_VERSION,
        "predictor": predictor,
        "sklearn_version": _get_sklearn_version(),
        "model_kind": getattr(predictor, "_model_kind", None),
    }
    with artifact.open("wb") as fh:
        pickle.dump(payload, fh)
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
    # sklearn pickles are not portable across versions — if the artifact was
    # written under a different sklearn than the one currently installed, reject
    # so the coordinator retrains under the running interpreter. When either
    # side is unknown (no sklearn at save time, or no sklearn now) we cannot
    # compare and let it through.
    artifact_sk = data.get("sklearn_version")
    installed_sk = _get_sklearn_version()
    if artifact_sk is not None and installed_sk is not None and artifact_sk != installed_sk:
        _LOGGER.warning(
            "Model artifact sklearn version drift: artifact=%s installed=%s — retraining",
            artifact_sk,
            installed_sk,
        )
        return None
    return data.get("predictor")


async def assemble_and_train(
    fetch_month: MonthFetcher,
    today: date,
    months: int = HISTORY_MONTHS_TARGET,
    executor: Executor | None = None,
) -> FuelPricePredictor:
    """Fetch `months` months via `fetch_month`, fit, and return the predictor.

    Raises RuntimeError if fewer than MIN_MONTHS_TO_TRAIN months yielded records.

    ``executor`` offloads the CPU-bound ``fit_predictor(series)`` call off the
    caller's context (default runs it inline via :func:`_default_executor`).
    The coordinator passes ``hass.async_add_executor_job`` so the fit does not
    block the HA event loop.
    """
    run = executor or _default_executor
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
    return await run(fit_predictor, series_from_records(records))
