import pickle
from datetime import date, timedelta

import pytest

from custom_components.fuel_predictor_wa.const import MIN_MONTHS_TO_TRAIN, MODEL_FILENAME
from custom_components.fuel_predictor_wa.trainer import (
    assemble_and_train,
    fit_predictor,
    load_model,
    save_model,
    series_from_records,
)


def test_series_collapses_to_min_price_per_day() -> None:
    d = date(2026, 7, 1)
    records = [
        {"date": d, "price": 190.0, "product": "ULP", "suburb": "A", "region": "R"},
        {"date": d, "price": 185.0, "product": "ULP", "suburb": "B", "region": "R"},
    ]
    assert series_from_records(records) == {d: 185.0}


def test_save_then_load_roundtrips(monkeypatch, tmp_path) -> None:
    series = {date(2026, 6, 1) + timedelta(days=i): 180.0 + (i % 7) for i in range(40)}
    predictor = fit_predictor(series)
    assert predictor._fitted  # noqa: SLF001

    artifact = save_model(predictor, tmp_path)
    assert artifact.name == MODEL_FILENAME

    loaded = load_model(artifact)
    assert loaded is not None
    # The average-baseline predictor exposed _overall_mean; the cycle-aware
    # HGBR model does not. Assert the round-trip via the still-stable public
    # contract: the loaded predictor is fitted and reports the same model_kind.
    assert loaded._fitted  # noqa: SLF001
    assert (
        loaded.train_metrics is not None
        and loaded.train_metrics["model_kind"] == predictor.train_metrics["model_kind"]
    )


async def _fake_fetch_all_ulp(year: int, month: int):
    """Fake fetcher returning one synthetic ULP record per call."""
    return [
        {
            "date": date(year, month, 1),
            "price": 180.0 + (month % 7),
            "product": "ULP",
            "suburb": "X",
            "region": "R",
        }
    ]


@pytest.mark.asyncio
async def test_assemble_and_train_succeeds_with_enough_months() -> None:
    predictor = await assemble_and_train(
        _fake_fetch_all_ulp, date(2026, 7, 1), months=MIN_MONTHS_TO_TRAIN + 1
    )
    assert predictor._fitted  # noqa: SLF001


@pytest.mark.asyncio
async def test_assemble_and_train_raises_when_too_few_months() -> None:
    async def fetch_none(year, month):
        return []

    with pytest.raises(RuntimeError):
        await assemble_and_train(fetch_none, date(2026, 7, 1), months=5)


def test_load_model_version_mismatch_returns_none(tmp_path) -> None:
    series = {date(2026, 1, 1): 180.0}
    predictor = fit_predictor(series)
    artifact = tmp_path / MODEL_FILENAME
    artifact.write_bytes(pickle.dumps({"version": 999, "predictor": predictor}))
    assert load_model(artifact) is None


def test_load_model_corrupt_bytes_returns_none(tmp_path) -> None:
    artifact = tmp_path / MODEL_FILENAME
    artifact.write_bytes(b"not a pickle - garbage bytes \x00\xff")
    assert load_model(artifact) is None
