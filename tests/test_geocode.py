"""Tests for suburb reverse-geocoding (nearest by site-bounds centroid)."""

from __future__ import annotations

from custom_components.fuel_predictor_wa.geocode import centroid_lon_lat, nearest_suburb


def _suburb(name: str, lon: float, lat: float) -> dict:
    return {
        "location": name,
        "postcode": 6000,
        "siteCount": 1,
        "siteBounds": {
            "type": "Polygon",
            "coordinates": [
                [
                    [lon, lat],
                    [lon + 0.01, lat],
                    [lon + 0.01, lat + 0.01],
                    [lon, lat + 0.01],
                    [lon, lat],
                ]
            ],
        },
    }


def test_centroid_of_polygon() -> None:
    bounds = {"type": "Polygon", "coordinates": [[[0, 0], [2, 0], [2, 2], [0, 2], [0, 0]]]}
    assert centroid_lon_lat(bounds) == (1.0, 1.0)


def test_centroid_none_for_missing_bounds() -> None:
    assert centroid_lon_lat(None) is None
    assert centroid_lon_lat({}) is None
    assert centroid_lon_lat({"type": "Polygon", "coordinates": []}) is None


def test_nearest_suburb_picks_closest() -> None:
    suburbs = [_suburb("BUNBURY", 115.64, -33.33), _suburb("PERTH", 115.86, -31.95)]
    assert nearest_suburb(suburbs, lat=-33.34, lon=115.64) == "BUNBURY"
    assert nearest_suburb(suburbs, lat=-31.96, lon=115.86) == "PERTH"


def test_nearest_suburb_ignores_entries_without_bounds() -> None:
    suburbs = [{"location": "NOWHERE", "siteBounds": None}, _suburb("BUNBURY", 115.64, -33.33)]
    assert nearest_suburb(suburbs, lat=-33.34, lon=115.64) == "BUNBURY"


def test_nearest_suburb_returns_none_if_all_invalid() -> None:
    assert nearest_suburb([{"location": "X", "siteBounds": None}], lat=0, lon=0) is None
