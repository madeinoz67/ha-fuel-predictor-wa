"""FuelWatch monthly historical-CSV helpers (Azure Blob).

Monthly files have a deterministic URL — no enumeration API is needed. Files are
NOT filterable, so callers stream/parse and filter client-side by product.
"""

from __future__ import annotations

import csv
import logging
from datetime import date
from io import StringIO
from pathlib import Path
from typing import Any

import aiohttp
from homeassistant.helpers.aiohttp_client import async_get_clientsession

from .const import BULK_CACHE_DIRNAME, HISTORIC_CSV_BASE, HISTORIC_CSV_TEMPLATE

_LOGGER = logging.getLogger(__name__)

_COL_DATE = "PUBLISH_DATE"
_COL_PRODUCT = "PRODUCT_DESCRIPTION"
_COL_PRICE = "PRODUCT_PRICE"
_COL_SUBURB = "LOCATION"
_COL_REGION = "REGION_DESCRIPTION"


def month_url(year: int, month: int) -> str:
    """Return the blob URL for a given month."""
    return f"{HISTORIC_CSV_BASE}/{HISTORIC_CSV_TEMPLATE.format(mm=month, yyyy=year)}"


def trailing_months(start: date, n: int) -> list[tuple[int, int]]:
    """Return n (year, month) tuples starting at `start`'s month and going back."""
    out: list[tuple[int, int]] = []
    y, m = start.year, start.month
    for _ in range(n):
        out.append((y, m))
        m -= 1
        if m == 0:
            m = 12
            y -= 1
    return out


def _parse_date(value: str) -> date:
    """Parse an Australian DD/MM/YYYY date."""
    dd, mm, yyyy = (int(x) for x in value.split("/"))
    return date(yyyy, mm, dd)


def parse_csv(text: str, product_description: str | None = None) -> list[dict[str, Any]]:
    """Parse a monthly CSV string into normalized records, optionally product-filtered."""
    reader = csv.DictReader(StringIO(text))
    records: list[dict[str, Any]] = []
    for row in reader:
        if product_description is not None and row.get(_COL_PRODUCT) != product_description:
            continue
        try:
            d = _parse_date(row[_COL_DATE])
            price = float(row[_COL_PRICE])
        except (KeyError, ValueError, TypeError):
            continue
        records.append(
            {
                "date": d,
                "price": price,
                "product": row.get(_COL_PRODUCT),
                "suburb": row.get(_COL_SUBURB),
                "region": row.get(_COL_REGION),
            }
        )
    return records


class HistoricClient:
    """Async fetcher for FuelWatch monthly historical CSVs."""

    def __init__(self, hass: Any) -> None:
        self._hass = hass
        self._session = async_get_clientsession(hass)

    async def async_fetch_month(
        self, year: int, month: int, product_description: str | None = None
    ) -> list[dict[str, Any]]:
        """Fetch one monthly CSV and return product-filtered records.

        Parsing (~40k rows) runs in the executor so the event loop is not blocked.
        """
        url = month_url(year, month)
        async with self._session.get(url, timeout=aiohttp.ClientTimeout(total=60)) as resp:
            resp.raise_for_status()
            text = await resp.text()
        return await self._hass.async_add_executor_job(parse_csv, text, product_description)


def _mutable_months(today: date) -> set[tuple[int, int]]:
    """The current and previous month — files that may still gain new days."""
    cur = (today.year, today.month)
    pm, py = today.month - 1, today.year
    if pm == 0:
        pm, py = 12, py - 1
    return {cur, (py, pm)}


WHOLESALE_TEMPLATE = "FuelWatchWholesale-{yyyy}.csv"


def parse_wholesale_csv(text: str, product_description: str | None = None) -> list[dict[str, Any]]:
    """Parse a FuelWatch *wholesale* (terminal-gate) monthly/yearly CSV.

    Same shape as the retail file except the product column is ``PRODUCT`` (not
    ``PRODUCT_DESCRIPTION``), so it gets its own parser. Records: {date, price,
    product, suburb}.
    """
    reader = csv.DictReader(StringIO(text))
    records: list[dict[str, Any]] = []
    for row in reader:
        if product_description is not None and row.get("PRODUCT") != product_description:
            continue
        try:
            d = _parse_date(row[_COL_DATE])
            price = float(row[_COL_PRICE])
        except (KeyError, ValueError, TypeError):
            continue
        records.append(
            {
                "date": d,
                "price": price,
                "product": row.get("PRODUCT"),
                "suburb": row.get(_COL_SUBURB),
            }
        )
    return records


async def fetch_wholesale_series(
    hass: Any,
    storage_dir: Path,
    product_description: str,
    years: int = 3,
) -> dict[date, float]:
    """WA terminal-gate price min/day for one product, from cached wholesale CSVs.

    The wholesale yearly CSVs (``FuelWatchWholesale-{yyyy}.csv``) sit on the same
    blob as the retail monthly files and carry the Singapore-Mogas-derived
    wholesale price WA retailers pay. Cached like the retail months.
    """
    cache_dir = Path(storage_dir) / BULK_CACHE_DIRNAME
    cache_dir.mkdir(parents=True, exist_ok=True)
    today = date.today()
    by_date: dict[date, float] = {}
    for offset in range(years):
        yr = today.year - offset
        cached = cache_dir / f"wholesale-{yr}.csv"
        mutable = offset == 0  # current year still gains days
        if cached.exists() and not mutable:
            text = await hass.async_add_executor_job(cached.read_text)
        else:
            url = f"{HISTORIC_CSV_BASE}/{WHOLESALE_TEMPLATE.format(yyyy=yr)}"
            session = async_get_clientsession(hass)
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=60)) as resp:
                resp.raise_for_status()
                text = await resp.text()
            await hass.async_add_executor_job(cached.write_text, text)
        for r in parse_wholesale_csv(text, product_description):
            d, p = r["date"], r["price"]
            if d not in by_date or p < by_date[d]:
                by_date[d] = p
    return by_date


async def async_fetch_month_cached(
    hass: Any,
    storage_dir: Path,
    year: int,
    month: int,
    product_description: str | None = None,
) -> list[dict[str, Any]]:
    """Fetch a month with on-disk caching: immutable months load from cache,
    the current + previous month re-download (they still gain days).

    Caches the raw CSV text (catchment-agnostic — the suburb filter is applied
    later at series collapse, so a catchment change reuses the cached months).
    """
    cache_dir = Path(storage_dir) / BULK_CACHE_DIRNAME
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / f"{year}-{month:02d}.csv"
    mutable = (year, month) in _mutable_months(date.today())

    def _read() -> str:
        return cached.read_text()

    def _write(text: str) -> None:
        cached.write_text(text)

    if cached.exists() and not mutable:
        text = await hass.async_add_executor_job(_read)
    else:
        session = async_get_clientsession(hass)
        url = month_url(year, month)
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=60)) as resp:
            resp.raise_for_status()
            text = await resp.text()
        await hass.async_add_executor_job(_write, text)
    return await hass.async_add_executor_job(parse_csv, text, product_description)
