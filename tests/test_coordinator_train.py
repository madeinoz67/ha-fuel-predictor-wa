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


async def _no_catchment() -> None:
    """Override for _async_resolve_catchment: no network in tests (trains WA-wide)."""
    return None


async def _no_tgp() -> None:
    """Override for _async_fetch_tgp: no wholesale fetch in tests (no drift term)."""
    return None


@pytest.mark.asyncio
async def test_first_refresh_trains_and_marks_ready(hass, tmp_path, monkeypatch) -> None:
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
    # Catchment resolution hits the FuelWatch suburbs API (network); override it
    # so the test trains WA-wide deterministically.
    monkeypatch.setattr(coord, "_async_resolve_catchment", _no_catchment)
    monkeypatch.setattr(coord, "_async_fetch_tgp", _no_tgp)

    # Stub the trainer so it does not hit the network.
    async def _fake_fetch(year, month):
        return [
            {
                "date": date(year, month, 1),
                "price": 180.0,
                "product": "ULP",
                "suburb": "BUNBURY",
                "region": "South West",
            }
        ]

    monkeypatch.setattr(coord, "_build_fetch_month", lambda: _fake_fetch)

    await coord.async_refresh()
    await hass.async_block_till_done()  # let the spawned trainer task finish

    assert coord.status == STATUS_READY
    assert coord.predictor._fitted  # noqa: SLF001
    assert (tmp_path / STORAGE_DIRNAME / _Entry.unique_id / "model.pkl").exists()

    # Teardown: cancel the coordinator's update_interval + debouncer so the
    # HA test harness does not flag a lingering timer after the test body.
    await coord.async_shutdown()


@pytest.mark.asyncio
async def test_predict_runs_in_executor(hass, tmp_path, monkeypatch) -> None:
    """predict() must run off the event loop via async_add_executor_job."""
    hass.config.config_dir = str(tmp_path)
    monkeypatch.setattr(
        "custom_components.fuel_predictor_wa.fuelwatch.async_get_clientsession",
        lambda _hass: None,
    )

    class _Entry:
        entry_id = "test-entry-exec"
        unique_id = "fuel_predictor_wa_exec"
        data = _entry_data()

    coord = FuelPredictorDataUpdateCoordinator(hass, _Entry())
    coord.client = _StubClient()
    monkeypatch.setattr(coord, "_async_resolve_catchment", _no_catchment)
    monkeypatch.setattr(coord, "_async_fetch_tgp", _no_tgp)

    async def _fake_fetch(year, month):
        return [
            {
                "date": date(year, month, 1),
                "price": 180.0,
                "product": "ULP",
                "suburb": "BUNBURY",
                "region": "South West",
            }
        ]

    monkeypatch.setattr(coord, "_build_fetch_month", lambda: _fake_fetch)

    # Wrap async_add_executor_job so we can observe what got offloaded.
    real_executor = hass.async_add_executor_job
    executor_calls: list = []

    async def _tracking_executor(func, *args, **kwargs):
        executor_calls.append(getattr(func, "__name__", repr(func)))
        return await real_executor(func, *args, **kwargs)

    monkeypatch.setattr(hass, "async_add_executor_job", _tracking_executor)

    await coord.async_refresh()
    await hass.async_block_till_done()

    # predict() ran through the executor during the refresh.
    assert "predict" in executor_calls, f"predict not offloaded; saw {executor_calls}"
    await coord.async_shutdown()


@pytest.mark.asyncio
async def test_setup_and_cancel_schedule(hass, tmp_path, monkeypatch) -> None:
    """The daily post-publication schedule registers + cleans up."""
    hass.config.config_dir = str(tmp_path)
    monkeypatch.setattr(
        "custom_components.fuel_predictor_wa.fuelwatch.async_get_clientsession",
        lambda _hass: None,
    )

    class _Entry:
        entry_id = "test-sched"
        unique_id = "fuel_predictor_wa_sched"
        data = _entry_data()

    coord = FuelPredictorDataUpdateCoordinator(hass, _Entry())
    assert coord._cancel_schedule is None  # noqa: SLF001
    coord.setup_schedule()
    assert coord._cancel_schedule is not None  # noqa: SLF001
    # Idempotent: a second setup does not replace the listener.
    first = coord._cancel_schedule  # noqa: SLF001
    coord.setup_schedule()
    assert coord._cancel_schedule is first  # noqa: SLF001
    coord.cancel_schedule()
    assert coord._cancel_schedule is None  # noqa: SLF001


class _StubClient:
    """Replaces FuelWatchClient so no live HTTP happens during the test."""

    async def async_fetch_today(self, *a, **k):
        return [{"price": 185.0, "brand": "Ampol", "location": "BUNBURY", "address": "1 Main St"}]

    async def async_fetch_tomorrow(self, *a, **k):
        return []
