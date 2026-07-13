"""Historical price handling: bulk CSV load + daily append + windowing.

The bulk CSV comes from data.wa.gov.au (see tools/download_history.py). Column
names are detected tolerantly — confirm them against the actual header on first
download and extend the candidate lists below if needed.
"""
from __future__ import annotations

import csv
import logging
from datetime import date, datetime
from pathlib import Path
from typing import Iterable

_LOGGER = logging.getLogger(__name__)

# Candidate column names (lowercased) for tolerant detection.
_DATE_COLS = ("date", "transactiondate", "price_date", "as_at_date", "pricingdate")
_PRICE_COLS = ("price", "fuel_price", "price_cents", "amount")
_PRODUCT_COLS = ("product", "product_description", "fuel_type", "productcode")
_SUBURB_COLS = ("suburb", "location", "town")
_SITE_COLS = ("site_id", "site-id", "site", "siteid")

_HISTORY_HEADER = ["date", "price", "product", "suburb", "site_id"]


def _find(row_keys: Iterable[str], candidates: tuple[str, ...]) -> str | None:
    lookup = {k.lower(): k for k in row_keys}
    for cand in candidates:
        if cand in lookup:
            return lookup[cand]
    return None


def load_history(path: str | Path) -> list[dict]:
    """Load a FuelWatch history CSV into normalized records.

    Each record: {date, price, product, suburb, site_id}. Rows with an
    unparseable date or price are skipped.
    """
    path = Path(path)
    if not path.exists():
        _LOGGER.warning("History file not found: %s", path)
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        cols = reader.fieldnames or []
        c_date = _find(cols, _DATE_COLS)
        c_price = _find(cols, _PRICE_COLS)
        c_product = _find(cols, _PRODUCT_COLS)
        c_suburb = _find(cols, _SUBURB_COLS)
        c_site = _find(cols, _SITE_COLS)
        if not (c_date and c_price):
            _LOGGER.error("History CSV missing date/price columns; found: %s", cols)
            return []
        rows: list[dict] = []
        for r in reader:
            try:
                d = datetime.fromisoformat(str(r[c_date])[:10]).date()
                p = float(r[c_price])
            except (ValueError, TypeError, KeyError):
                continue
            rows.append(
                {
                    "date": d,
                    "price": p,
                    "product": (r.get(c_product) if c_product else None),
                    "suburb": (r.get(c_suburb) if c_suburb else None),
                    "site_id": (r.get(c_site) if c_site else None),
                }
            )
        _LOGGER.info("Loaded %d history rows from %s", len(rows), path)
        return rows


def window(rows: list[dict], days: int, product: str | None = None) -> dict[date, float]:
    """Reduce history to {date: min_price_for_product} over the last `days` days."""
    by_date: dict[date, float] = {}
    for r in rows:
        if product is not None and str(r.get("product")) != str(product):
            continue
        d = r["date"]
        price = r["price"]
        if d not in by_date or price < by_date[d]:
            by_date[d] = price
    if not by_date:
        return {}
    latest = max(by_date)
    cutoff = latest.fromordinal(latest.toordinal() - days + 1)
    return {d: p for d, p in by_date.items() if d >= cutoff}


def append_daily(path: str | Path, entries: Iterable[dict], today: date) -> int:
    """Append today's polled entries to the local history CSV (creates if missing).

    Each entry should have: price, product, suburb, site_id (plus optional brand).
    Returns the number of rows appended.
    """
    path = Path(path)
    new_header = not path.exists()
    n = 0
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        if new_header:
            writer.writerow(_HISTORY_HEADER)
        for e in entries:
            writer.writerow(
                [
                    today.isoformat(),
                    e.get("price"),
                    e.get("product"),
                    e.get("location") or e.get("suburb"),
                    e.get("site_id"),
                ]
            )
            n += 1
    return n
