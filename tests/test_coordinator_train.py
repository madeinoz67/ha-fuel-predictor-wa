from datetime import date

import pytest

from custom_components.fuel_predictor_wa.const import (
    PRODUCT_UNLEADED,
    STATUS_READY,
    STORAGE_DIRNAME,
)
from custom_components.fuel_predictor_wa.coordinator import (
    FuelPredictorDataUpdateCoordinator,
)


def _entry_data():
    return {
        "product": PRODUCT_UNLEADED,
        "suburb": "BUNBURY",
        "surrounding": True,
        "forecast_horizon_days": 7,
        "station_limit": 5,
    }


@pytest.mark.asyncio
async def test_first_refresh_trains_and_marks_ready(
    hass, tmp_path, monkeypatch
) -> None:
    """With no artifact and a stubbed fetch, first refresh ends status=ready."""
    hass.config.config_dir = str(tmp_path)

    # FuelWatchClient.__init__ calls async_get_clientsession(hass), which
    # initialises aiodns/pycares and spawns a DNS-resolver thread the HA test
    # harness flags at teardown. We replace coord.client with _StubClient()
    # below, so the session is never used — stub the session factory to keep
    # the harness clean without touching production code or weakening any
    # coordinator assertion.
    monkeypatch.setattr(
        "custom_components.fuel_predictor_wa.fuelwatch.async_get_clientsession",
        lambda _hass: None,
    )

    class _Entry:
        entry_id = "test-entry"
        unique_id = "fuel_predictor_wa_1_bunbury"
        data = _entry_data()

    coord = FuelPredictorDataUpdateCoordinator(hass, _Entry())
    coord.client = _StubClient()  # no live network

    # Stub the trainer so it does not hit the network.
    async def _fake_fetch(year, month):
        return [{"date": date(year, month, 1), "price": 180.0,
                 "product": "ULP", "suburb": "BUNBURY", "region": "South West"}]

    monkeypatch.setattr(
        coord, "_build_fetch_month", lambda: _fake_fetch
    )

    await coord.async_refresh()
    await hass.async_block_till_done()  # let the spawned trainer task finish

    assert coord.status == STATUS_READY
    assert coord.predictor._fitted  # noqa: SLF001
    assert (tmp_path / STORAGE_DIRNAME / _Entry.unique_id / "model.pkl").exists()

    # Teardown: cancel the coordinator's update_interval + debouncer so the
    # HA test harness does not flag a lingering timer after the test body.
    await coord.async_shutdown()


class _StubClient:
    """Replaces FuelWatchClient so no live HTTP happens during the test."""

    async def async_fetch_today(self, *a, **k):
        return [{"price": 185.0, "brand": "Ampol", "location": "BUNBURY",
                 "address": "1 Main St"}]

    async def async_fetch_tomorrow(self, *a, **k):
        return []
