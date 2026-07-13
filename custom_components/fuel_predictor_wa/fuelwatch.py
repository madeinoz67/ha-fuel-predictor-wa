"""Async FuelWatch client (today/tomorrow only).

The public FuelWatch service exposes the same data as the RSS feed
(https://www.fuelwatch.wa.gov.au/tools/rss) — daily prices fixed per day; only
``today`` and ``tomorrow`` (lodged by ~14:00 AWST) are available.
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from typing import Any

from homeassistant.helpers.aiohttp_client import async_get_clientsession

from .const import FUELWATCH_DAY_TODAY, FUELWATCH_DAY_TOMORROW, FUELWATCH_ENDPOINT

_LOGGER = logging.getLogger(__name__)

# FuelWatch XML child tags (hyphenated) → friendly keys.
_SITE_FIELDS = {
    "price": "price",
    "brand": "brand",
    "trading-name": "trading_name",
    "location": "location",
    "address": "address",
    "phone": "phone",
    "site-id": "site_id",
    "latitude": "latitude",
    "longitude": "longitude",
    "product": "product",
}


class FuelWatchClient:
    """Async wrapper around the public FuelWatch XML service."""

    def __init__(self, hass: Any) -> None:
        self._session = async_get_clientsession(hass)

    async def async_fetch(
        self, product: int, suburb: str, day: str, surrounding: bool = True
    ) -> list[dict[str, Any]]:
        """Fetch station prices for a product/suburb/day."""
        params = {
            "product": str(product),
            "suburb": suburb,
            "day": day,
            "surrounding": "yes" if surrounding else "no",
        }
        async with self._session.get(FUELWATCH_ENDPOINT, params=params, timeout=30) as resp:
            resp.raise_for_status()
            xml_text = await resp.text()
        return self.parse(xml_text)

    async def async_fetch_today(
        self, product: int, suburb: str, surrounding: bool = True
    ) -> list[dict[str, Any]]:
        return await self.async_fetch(product, suburb, FUELWATCH_DAY_TODAY, surrounding)

    async def async_fetch_tomorrow(
        self, product: int, suburb: str, surrounding: bool = True
    ) -> list[dict[str, Any]]:
        return await self.async_fetch(product, suburb, FUELWATCH_DAY_TOMORROW, surrounding)

    @staticmethod
    def parse(xml_text: str) -> list[dict[str, Any]]:
        """Parse FuelWatch XML (site elements or RSS items) → station dicts.

        Robust to both the SOAP-style ``<site>`` and RSS ``<item>`` shapes.
        """
        # FuelWatch RSS is served with a UTF-8 BOM; strip it or ET.fromstring fails.
        xml_text = xml_text.lstrip("﻿")
        root = ET.fromstring(xml_text)
        sites = list(root.iter("site")) or list(root.iter("item"))
        parsed: list[dict[str, Any]] = []
        for site_el in sites:
            entry: dict[str, Any] = {}
            for child in site_el:
                tag = child.tag.split("}")[-1]
                if tag in _SITE_FIELDS:
                    entry[_SITE_FIELDS[tag]] = child.text
            if "price" not in entry:
                continue
            try:
                entry["price"] = float(entry["price"])  # type: ignore[arg-type]
            except (TypeError, ValueError):
                _LOGGER.debug("Skipping site with non-numeric price: %s", entry)
                continue
            parsed.append(entry)
        return parsed
