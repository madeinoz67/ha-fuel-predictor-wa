"""Yahoo Finance client for global leading-indicator prices.

Fetches daily close prices for the symbols used by the predictor's
global-feature layer: RBOB gasoline (RB=F), Brent crude (BZ=F), and AUD/USD
(AUDUSD=X). Global wholesale prices lead WA retail by ~1-2 weeks (pass-through
through the terminal gate price + retail margin), so they are useful leading
indicators at the configured lag (see ``GLOBAL_FEATURE_LAG_DAYS``).

The Yahoo chart API is key-less; it only requires a ``User-Agent`` header. Each
symbol is fetched independently — a single symbol failing (network error, 429,
malformed payload) is logged and skipped so one bad ticker can never poison the
whole feature set.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, date, datetime
from typing import TYPE_CHECKING
from urllib.parse import quote

import aiohttp
from homeassistant.helpers.aiohttp_client import async_get_clientsession

from .const import YAHOO_BASE_URL, YAHOO_SYMBOLS

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

_LOGGER = logging.getLogger(__name__)

# Reported in the User-Agent header so Yahoo can identify and rate-limit this
# integration rather than blocking the shared HA session outright.
_USER_AGENT = "ha-fuel-predictor-wa/0.1 (+https://github.com/madeinoz67/ha-fuel-predictor-wa)"
_REQUEST_TIMEOUT = 30  # seconds per symbol request


def parse_chart(json_text: str) -> dict[date, float]:
    """Parse one symbol's Yahoo chart JSON text into ``{date: close}``.

    Pure (no network) so it is unit-testable directly. Returns ``{}`` for any
    malformed payload: non-JSON text, a non-null chart-level ``error``, a missing
    ``result`` list, or missing ``timestamp``/``indicators.quote[0].close`` arrays.
    Null close values are skipped (Yahoo returns ``null`` for non-trading days).
    """
    try:
        payload = json.loads(json_text)
    except (json.JSONDecodeError, TypeError):
        return {}

    chart = payload.get("chart") if isinstance(payload, dict) else None
    if not isinstance(chart, dict):
        return {}

    if chart.get("error") is not None:
        return {}

    result = chart.get("result")
    if not result or not isinstance(result, list):
        return {}

    entry = result[0]
    if not isinstance(entry, dict):
        return {}

    timestamps = entry.get("timestamp")
    quote = entry.get("indicators", {}).get("quote")
    if not isinstance(timestamps, list) or not isinstance(quote, list) or not quote:
        return {}

    closes = quote[0].get("close") if isinstance(quote[0], dict) else None
    if not isinstance(closes, list):
        return {}

    out: dict[date, float] = {}
    for ts, close in zip(timestamps, closes, strict=False):
        if close is None:
            continue
        try:
            d = datetime.fromtimestamp(int(ts), tz=UTC).date()
            out[d] = float(close)
        except (TypeError, ValueError, OverflowError, OSError):
            continue
    return out


class GlobalFeaturesClient:
    """Async fetcher for Yahoo Finance daily close prices."""

    def __init__(self, hass: HomeAssistant) -> None:
        self._hass = hass
        self._session = async_get_clientsession(hass)

    async def _fetch(
        self, symbols: tuple[str, ...], range_str: str
    ) -> dict[str, dict[date, float]]:
        """Fetch close history for each symbol, skipping per-symbol failures."""
        results: dict[str, dict[date, float]] = {}
        for symbol in symbols:
            url = f"{YAHOO_BASE_URL}/{quote(symbol, safe='')}?range={range_str}&interval=1d"
            try:
                async with self._session.get(
                    url,
                    headers={"User-Agent": _USER_AGENT},
                    timeout=aiohttp.ClientTimeout(total=_REQUEST_TIMEOUT),
                ) as resp:
                    resp.raise_for_status()
                    text = await resp.text()
            except (aiohttp.ClientError, TimeoutError) as err:
                _LOGGER.warning("Yahoo Finance fetch failed for %s: %s", symbol, err)
                continue
            except Exception as err:  # noqa: BLE001 — never let one symbol kill the batch
                _LOGGER.warning("Yahoo Finance fetch errored for %s: %s", symbol, err)
                continue

            parsed = parse_chart(text)
            if not parsed:
                _LOGGER.warning("Yahoo Finance returned no usable data for %s", symbol)
                continue
            results[symbol] = parsed
        return results

    async def async_fetch_history(
        self,
        symbols: tuple[str, ...] = YAHOO_SYMBOLS,
        range_str: str = "2y",
    ) -> dict[str, dict[date, float]]:
        """Fetch long-range daily closes (default 2 years) per symbol.

        Failed symbols are skipped; if all fail, returns ``{}``.
        """
        return await self._fetch(symbols, range_str)

    async def async_fetch_recent(
        self,
        symbols: tuple[str, ...] = YAHOO_SYMBOLS,
        range_str: str = "1mo",
    ) -> dict[str, dict[date, float]]:
        """Fetch short-range daily closes (default 1 month) per symbol.

        Used at predict time for the most recent global values. Failed symbols
        are skipped; if all fail, returns ``{}``.
        """
        return await self._fetch(symbols, range_str)


__all__: list[str] = ["GlobalFeaturesClient", "parse_chart"]
