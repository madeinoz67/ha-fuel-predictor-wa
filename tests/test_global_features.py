"""Tests for the Yahoo Finance global leading-indicators client."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime

import pytest

from custom_components.fuel_predictor_wa.global_features import (
    GlobalFeaturesClient,
    parse_chart,
)

# Three consecutive UTC days starting 2023-11-14 22:50:40Z.
_TIMESTAMPS = [1700000000 + i * 86400 for i in range(3)]


def _expected_date(i: int) -> date:
    return datetime.fromtimestamp(_TIMESTAMPS[i], tz=UTC).date()


def _chart_json(closes: list[float | None], symbol: str = "RB=F") -> str:
    """Build a minimal but well-formed Yahoo chart payload."""
    return json.dumps(
        {
            "chart": {
                "result": [
                    {
                        "meta": {"symbol": symbol, "regularMarketPrice": closes[-1]},
                        "timestamp": _TIMESTAMPS[: len(closes)],
                        "indicators": {"quote": [{"close": closes}]},
                    }
                ],
                "error": None,
            }
        }
    )


# --- parse_chart (pure, no network) ---


def test_parse_chart_builds_date_to_close() -> None:
    payload = _chart_json([2.10, None, 2.25])
    result = parse_chart(payload)
    # null close at index 1 is skipped
    assert result == {
        _expected_date(0): 2.10,
        _expected_date(2): 2.25,
    }


def test_parse_chart_handles_error_payload() -> None:
    payload = json.dumps({"chart": {"result": None, "error": "Invalid symbol"}})
    assert parse_chart(payload) == {}


def test_parse_chart_handles_missing_result() -> None:
    payload = json.dumps({"chart": {}})
    assert parse_chart(payload) == {}


def test_parse_chart_handles_malformed_json() -> None:
    assert parse_chart("not json at all") == {}


def test_parse_chart_handles_missing_quote() -> None:
    # result present but indicators/quote absent -> graceful empty
    payload = json.dumps(
        {
            "chart": {
                "result": [{"meta": {}, "timestamp": _TIMESTAMPS, "indicators": {}}],
                "error": None,
            }
        }
    )
    assert parse_chart(payload) == {}


# --- async client (fake session, no real network) ---


class _FakeResp:
    def __init__(self, payload: str) -> None:
        self._payload = payload

    async def __aenter__(self) -> _FakeResp:
        return self

    async def __aexit__(self, *exc) -> bool:
        return False

    def raise_for_status(self) -> None:
        return None

    async def text(self) -> str:
        return self._payload

    async def json(self) -> dict:
        return json.loads(self._payload)


class _RaisingResp:
    """Simulates a 429 / network failure when entered + status-checked."""

    async def __aenter__(self) -> _RaisingResp:
        return self

    async def __aexit__(self, *exc) -> bool:
        return False

    def raise_for_status(self) -> None:
        msg = "429 Too Many Requests"
        raise RuntimeError(msg)


class _FakeSession:
    """Good for RB=F and AUDUSD=X, raises for BZ=F."""

    def __init__(self) -> None:
        self.urls: list[str] = []

    def get(self, url: str, **kwargs):  # noqa: ANN001, ANN201
        self.urls.append(url)
        if "RB%3DF" in url:
            return _FakeResp(_chart_json([2.0, 2.1, 2.2], symbol="RB=F"))
        if "BZ%3DF" in url:
            return _RaisingResp()
        if "AUDUSD%3DX" in url:
            return _FakeResp(_chart_json([0.65, 0.66, 0.67], symbol="AUDUSD=X"))
        msg = f"unexpected url: {url}"
        raise AssertionError(msg)


class _FakeHass:
    """Minimal hass stub — the client only stores it."""


@pytest.mark.asyncio
async def test_async_fetch_history_skips_failed_symbols() -> None:
    client = GlobalFeaturesClient.__new__(GlobalFeaturesClient)
    client._session = _FakeSession()  # type: ignore[attr-defined]
    client._hass = _FakeHass()  # type: ignore[attr-defined]

    result = await client.async_fetch_history(symbols=("RB=F", "BZ=F", "AUDUSD=X"))

    # RB=F and AUDUSD=X succeeded; BZ=F failed and was skipped.
    assert "RB=F" in result
    assert "AUDUSD=X" in result
    assert "BZ=F" not in result

    # The successful payload has 3 closes (no nulls) -> 3 date entries each.
    assert len(result["RB=F"]) == 3
    assert len(result["AUDUSD=X"]) == 3
    assert result["RB=F"][_expected_date(2)] == 2.2


@pytest.mark.asyncio
async def test_async_fetch_history_all_fail_returns_empty() -> None:
    class _AllFail:
        def get(self, url: str, **kwargs):  # noqa: ANN001, ANN201, ARG002
            return _RaisingResp()

    client = GlobalFeaturesClient.__new__(GlobalFeaturesClient)
    client._session = _AllFail()  # type: ignore[attr-defined]
    client._hass = _FakeHass()  # type: ignore[attr-defined]

    result = await client.async_fetch_history(symbols=("RB=F", "BZ=F"))
    assert result == {}
