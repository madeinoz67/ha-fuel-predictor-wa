"""Unit tests for the local-catchment resolver (no HA runtime needed)."""

from __future__ import annotations

import json

from custom_components.fuel_predictor_wa.catchment import (
    catchment_key,
    is_cache_fresh,
    load_cached_catchment,
    resolve_catchment,
    save_cached_catchment,
)


def _box(lon: float, lat: float) -> dict:
    """A tiny GeoJSON Polygon box around (lon, lat) — centroid == (lon, lat)."""
    d = 0.001
    ring = [
        [lon - d, lat - d],
        [lon + d, lat - d],
        [lon + d, lat + d],
        [lon - d, lat + d],
        [lon - d, lat - d],
    ]
    return {"type": "Polygon", "coordinates": [ring]}


def _sub(name: str, lon: float, lat: float, count: int) -> dict:
    return {"location": name, "siteCount": count, "siteBounds": _box(lon, lat)}


# Bunbury ~(-33.33, 115.63); nearby suburbs; Perth far north.
SUBURBS = [
    _sub("BUNBURY", 115.63, -33.33, 4),
    _sub("EAST BUNBURY", 115.65, -33.32, 3),
    _sub("CAREY PARK", 115.64, -33.35, 3),
    _sub("EATON", 115.70, -33.30, 5),
    _sub("PERTH", 115.86, -31.95, 500),
]


def test_expands_to_min_stations_with_nearest_first() -> None:
    c = resolve_catchment(SUBURBS, "Bunbury", min_stations=10)
    assert c is not None
    names = set(c["suburbs"])
    assert names == {"BUNBURY", "EAST BUNBURY", "CAREY PARK"}
    assert c["total_stations"] == 10  # 4 + 3 + 3
    assert "PERTH" not in names  # far away, not needed
    # Members ranked nearest-first; anchor at distance 0.
    assert c["members"][0]["name"] == "BUNBURY"
    assert c["members"][0]["distance_km"] == 0.0


def test_anchor_already_meets_gate_returns_just_anchor() -> None:
    c = resolve_catchment(SUBURBS, "Bunbury", min_stations=2)
    assert c is not None
    assert c["suburbs"] == ["BUNBURY"]
    assert c["total_stations"] == 4


def test_anchor_suburb_case_insensitive() -> None:
    c = resolve_catchment(SUBURBS, "  bunbury  ", min_stations=5)
    assert c is not None
    assert c["anchor"] == "BUNBURY"


def test_unknown_anchor_returns_none() -> None:
    assert resolve_catchment(SUBURBS, "NOWHERE", min_stations=10) is None


def test_keeps_expanding_when_anchor_under_gate() -> None:
    # Anchor alone (4) < 9 -> must include the next nearest until >= 9.
    c = resolve_catchment(SUBURBS, "Bunbury", min_stations=9)
    assert c is not None
    assert c["total_stations"] >= 9
    assert "BUNBURY" in c["suburbs"]


def test_catchment_key_changes_with_suburb_set() -> None:
    a = resolve_catchment(SUBURBS, "Bunbury", min_stations=10)
    b = resolve_catchment(SUBURBS, "Bunbury", min_stations=2)
    assert a is not None and b is not None
    assert catchment_key(2, a) != catchment_key(2, b)


def test_cache_round_trip_and_freshness(tmp_path) -> None:
    c = resolve_catchment(SUBURBS, "Bunbury", min_stations=10)
    assert c is not None
    save_cached_catchment(tmp_path, c)
    loaded = load_cached_catchment(tmp_path)
    assert loaded is not None
    assert set(loaded["suburbs"]) == {"BUNBURY", "EAST BUNBURY", "CAREY PARK"}
    assert is_cache_fresh(loaded, product=2, suburb="Bunbury", min_stations=10)
    # Stale if the config changed.
    assert not is_cache_fresh(loaded, product=2, suburb="Bunbury", min_stations=40)
    assert not is_cache_fresh(loaded, product=2, suburb="Perth", min_stations=10)


def test_save_none_clears_cache(tmp_path) -> None:
    c = resolve_catchment(SUBURBS, "Bunbury", min_stations=10)
    assert c is not None
    save_cached_catchment(tmp_path, c)
    save_cached_catchment(tmp_path, None)
    assert load_cached_catchment(tmp_path) is None


def test_load_corrupt_cache_returns_none(tmp_path) -> None:
    (tmp_path / "catchment.json").write_text("not json")
    assert load_cached_catchment(tmp_path) is None


def test_cache_file_is_json_serializable(tmp_path) -> None:
    c = resolve_catchment(SUBURBS, "Bunbury", min_stations=10)
    assert c is not None
    save_cached_catchment(tmp_path, c)
    # Must be valid JSON with the members list intact.
    data = json.loads((tmp_path / "catchment.json").read_text())
    assert data["anchor"] == "BUNBURY"
    assert len(data["members"]) == 3
