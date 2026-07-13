"""Reverse-geocode Home Assistant's location to the nearest FuelWatch suburb.

Uses the FuelWatch /api/sites/suburbs list (each suburb carries a GeoJSON
Polygon over its station sites) and picks the suburb whose centroid is nearest
to HA's configured location. No external geocoder.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from homeassistant.core import HomeAssistant
from homeassistant.helpers.aiohttp_client import async_get_clientsession

from .const import FUELWATCH_SUBURBS_ENDPOINT

_LOGGER = logging.getLogger(__name__)


def centroid_lon_lat(bounds: dict[str, Any] | None) -> tuple[float, float] | None:
    """Return (lon, lat) centroid of a GeoJSON Polygon's outer ring, or None."""
    if not isinstance(bounds, dict):
        return None
    try:
        ring = bounds["coordinates"][0]
    except (KeyError, IndexError, TypeError):
        return None
    if not ring:
        return None
    # GeoJSON rings are closed (first == last); drop the duplicate closing
    # vertex so it isn't double-counted in the average.
    if len(ring) > 1 and ring[0] == ring[-1]:
        ring = ring[:-1]
    lons = [p[0] for p in ring]
    lats = [p[1] for p in ring]
    return sum(lons) / len(lons), sum(lats) / len(lats)


def nearest_suburb(suburbs: list[dict[str, Any]], lat: float, lon: float) -> str | None:
    """Pick the suburb whose site-bounds centroid is nearest to (lat, lon)."""
    best: tuple[float, str] | None = None
    for suburb in suburbs:
        centroid = centroid_lon_lat(suburb.get("siteBounds"))
        if centroid is None:
            continue
        c_lon, c_lat = centroid
        # Squared equirectangular distance — fine for ranking nearby suburbs.
        dist = (c_lat - lat) ** 2 + (c_lon - lon) ** 2
        name = suburb.get("location")
        if not name:
            continue
        if best is None or dist < best[0]:
            best = (dist, str(name))
    return best[1] if best else None


async def async_detect_suburb(hass: HomeAssistant) -> str | None:
    """Detect the nearest FuelWatch suburb to HA's configured location.

    Best-effort: any failure (no location, network, slow API, bad payload)
    returns None so the config flow always renders. A hard 5 s guard prevents a
    slow/unreachable FuelWatch API from hanging the flow (which HA surfaces to
    the user as a 500).
    """
    lat = hass.config.latitude
    lon = hass.config.longitude
    if lat is None or lon is None:
        _LOGGER.debug("HA location not configured; cannot auto-detect suburb")
        return None
    session = async_get_clientsession(hass)
    try:
        async with asyncio.timeout(5):
            async with session.get(FUELWATCH_SUBURBS_ENDPOINT) as resp:
                resp.raise_for_status()
                suburbs = await resp.json()
            return nearest_suburb(suburbs, lat, lon)
    except Exception as err:  # noqa: BLE001 — best-effort; never break the config flow
        _LOGGER.debug("Could not auto-detect suburb: %s", err)
        return None
