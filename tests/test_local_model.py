"""Unit tests for the local-model data path (catchment filter + month cache)."""

from __future__ import annotations

from datetime import date

from custom_components.fuel_predictor_wa.historic_client import _mutable_months
from custom_components.fuel_predictor_wa.trainer import series_from_records


def test_series_filter_excludes_outside_catchment_and_keeps_min_per_day() -> None:
    records = [
        {"date": date(2026, 7, 1), "price": 170.0, "suburb": "BUNBURY"},
        {"date": date(2026, 7, 1), "price": 165.0, "suburb": "EAST BUNBURY"},
        {"date": date(2026, 7, 1), "price": 150.0, "suburb": "PERTH"},  # excluded
        {"date": date(2026, 7, 2), "price": 168.0, "suburb": "BUNBURY"},
    ]
    s = series_from_records(records, suburbs_filter={"Bunbury", "East Bunbury"})
    assert s == {date(2026, 7, 1): 165.0, date(2026, 7, 2): 168.0}


def test_series_filter_is_case_insensitive() -> None:
    records = [{"date": date(2026, 7, 1), "price": 170.0, "suburb": "Bunbury"}]
    assert series_from_records(records, suburbs_filter={"BUNBURY"}) == {date(2026, 7, 1): 170.0}


def test_series_no_filter_keeps_all() -> None:
    records = [{"date": date(2026, 7, 1), "price": 150.0, "suburb": "PERTH"}]
    assert series_from_records(records) == {date(2026, 7, 1): 150.0}


def test_series_filter_takes_min_within_catchment() -> None:
    records = [
        {"date": date(2026, 7, 1), "price": 170.0, "suburb": "BUNBURY"},
        {"date": date(2026, 7, 1), "price": 168.0, "suburb": "EATON"},
    ]
    s = series_from_records(records, suburbs_filter={"BUNBURY", "EATON"})
    assert s[date(2026, 7, 1)] == 168.0


def test_mutable_months_current_and_previous() -> None:
    mut = _mutable_months(date(2026, 7, 14))
    assert (2026, 7) in mut
    assert (2026, 6) in mut
    assert (2026, 5) not in mut


def test_mutable_months_wraps_year_boundary() -> None:
    mut = _mutable_months(date(2026, 1, 5))
    assert (2026, 1) in mut
    assert (2025, 12) in mut
