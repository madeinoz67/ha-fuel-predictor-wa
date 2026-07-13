"""FuelWatch monthly historical-CSV helpers (Azure Blob).

Monthly files have a deterministic URL — no enumeration API is needed. Files are
NOT filterable, so callers stream/parse and filter client-side by product.
"""
from __future__ import annotations

import csv
import logging
from datetime import date
from io import StringIO
from typing import Any

from .const import HISTORIC_CSV_BASE, HISTORIC_CSV_TEMPLATE

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
