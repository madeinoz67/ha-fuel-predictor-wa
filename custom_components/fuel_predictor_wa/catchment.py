"""Resolve a LOCAL training catchment for the configured suburb.

The model trains on too-wide data by default (WA-wide min/day, which is
Perth-skewed — regional users see a real accuracy loss). This module defines the
catchment the trainer filters to: the configured suburb plus the nearest suburbs
by centroid distance, expanded until the cumulative station count reaches a
minimum (default 40 — the regional sweet spot from the accuracy spike).

Data source: FuelWatch ``/api/sites/suburbs`` returns every suburb with
``siteCount`` (station count) and ``siteBounds`` (a GeoJSON Polygon whose centroid
we already compute via :func:`geocode.centroid_lon_lat`). So the gate needs no
extra geocoder — just sort by distance and accumulate ``siteCount``.

The resolver is pure (operates on the fetched suburbs list); the coordinator
fetches + caches the result so a restart or a 24h refit doesn't re-hit the API.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
from datetime import date
from pathlib import Path
from typing import Any

from homeassistant.core import HomeAssistant
from homeassistant.helpers.aiohttp_client import async_get_clientsession

from .const import CATCHMENT_FILENAME, FUELWATCH_SUBURBS_ENDPOINT
from .geocode import centroid_lon_lat

_LOGGER = logging.getLogger(__name__)


def _haversine_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Great-circle distance in km (for human-readable metadata only)."""
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def resolve_catchment(
    suburbs: list[dict[str, Any]],
    suburb: str,
    min_stations: int,
) -> dict[str, Any] | None:
    """Resolve the catchment for ``suburb`` from the FuelWatch suburbs list.

    Returns ``{suburbs: set[str], total_stations, members: [{name, site_count,
    distance_km}], anchor}``, or ``None`` if the configured suburb isn't in the
    list (caller falls back to WA-wide). Expands from the anchor outward by
    centroid distance until cumulative ``siteCount >= min_stations``.
    """
    norm = suburb.strip().upper()
    anchor = next((s for s in suburbs if str(s.get("location", "")).upper() == norm), None)
    if anchor is None:
        return None
    anchor_c = centroid_lon_lat(anchor.get("siteBounds"))
    if anchor_c is None:
        return None

    # Rank every suburb by distance from the anchor centroid.
    ranked: list[tuple[float, dict[str, Any], tuple[float, float]]] = []
    for s in suburbs:
        c = centroid_lon_lat(s.get("siteBounds"))
        name = s.get("location")
        if c is None or not name:
            continue
        # Equirectangular squared distance is fine for ranking; convert to km
        # only for the metadata.
        d2 = (c[1] - anchor_c[1]) ** 2 + (c[0] - anchor_c[0]) ** 2
        ranked.append((d2, s, c))
    ranked.sort(key=lambda t: t[0])

    members: list[dict[str, Any]] = []
    total = 0
    catchment: set[str] = set()
    for _d2, s, c in ranked:
        name = str(s["location"])
        site_count = int(s.get("siteCount", 0) or 0)
        km = _haversine_km(anchor_c[0], anchor_c[1], c[0], c[1])
        members.append({"name": name, "site_count": site_count, "distance_km": round(km, 1)})
        catchment.add(name)
        total += site_count
        if total >= min_stations:
            break

    return {
        "anchor": str(anchor["location"]),
        "suburbs": sorted(catchment),
        "total_stations": total,
        "min_stations": min_stations,
        "members": members,
    }


async def async_fetch_suburbs(hass: HomeAssistant) -> list[dict[str, Any]] | None:
    """Fetch the FuelWatch suburbs list (best-effort, 8 s guard)."""
    session = async_get_clientsession(hass)
    try:
        async with asyncio.timeout(8):
            async with session.get(FUELWATCH_SUBURBS_ENDPOINT) as resp:
                resp.raise_for_status()
                data = await resp.json()
            return data if isinstance(data, list) else None
    except Exception as err:  # noqa: BLE001 — best-effort; never break setup
        _LOGGER.debug("Could not fetch FuelWatch suburbs list: %s", err)
        return None


def catchment_key(product: int, catchment: dict[str, Any] | None) -> str:
    """Stable cache key: changes when product or the resolved suburb set changes."""
    if catchment is None:
        return f"p{product}_WA"
    suburbs = "|".join(catchment["suburbs"])
    return f"p{product}_{catchment['anchor']}_{catchment['total_stations']}_{suburbs}"


def load_cached_catchment(storage_dir: Path) -> dict[str, Any] | None:
    """Load the last resolved catchment from disk (None if absent/corrupt)."""
    path = storage_dir / CATCHMENT_FILENAME
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def save_cached_catchment(storage_dir: Path, catchment: dict[str, Any] | None) -> None:
    """Persist the resolved catchment (None clears it)."""
    storage_dir.mkdir(parents=True, exist_ok=True)
    path = storage_dir / CATCHMENT_FILENAME
    if catchment is None:
        path.unlink(missing_ok=True)
        return
    # Attach a resolved date so staleness is visible (not used for correctness).
    payload = {**catchment, "resolved_date": date.today().isoformat()}
    path.write_text(json.dumps(payload))


def is_cache_fresh(
    cached: dict[str, Any] | None,
    product: int,
    suburb: str,
    min_stations: int,
) -> bool:
    """True if the cached catchment still matches the current config."""
    if not cached:
        return False
    return (
        cached.get("anchor", "").upper() == suburb.strip().upper()
        and cached.get("min_stations") == min_stations
        and catchment_key(product, cached) == catchment_key(product, cached)
    )
