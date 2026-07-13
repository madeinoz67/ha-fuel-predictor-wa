"""Tests for the FuelWatch RSS client parser.

fuelwatch.py had no test in the scaffold, which let the wrong-endpoint/BOM bug
slip through. These pin the real RSS shape (BOM + <item>) so it can't regress.
"""

from __future__ import annotations

from custom_components.fuel_predictor_wa.fuelwatch import FuelWatchClient

# Real-shape FuelWatch RSS feed: leading UTF-8 BOM + one abridged <item>.
RSS = (
    '﻿<rss version="2.0"><channel><title>FuelWatch Prices For Bunbury</title>'
    "<item>"
    "<title>161.3: Vibe Bunbury South</title>"
    "<brand>Vibe</brand><date>2026-07-13</date><price>161.3</price>"
    "<trading-name>Vibe Bunbury South</trading-name>"
    "<location>SOUTH BUNBURY</location><address>302 Blair St</address>"
    "<latitude>-33.351866</latitude><longitude>115.641941</longitude>"
    "</item></channel></rss>"
)


def test_parse_rss_item_strips_bom_and_normalises() -> None:
    sites = FuelWatchClient.parse(RSS)
    assert len(sites) == 1
    s = sites[0]
    assert s["price"] == 161.3
    assert s["brand"] == "Vibe"
    assert s["location"] == "SOUTH BUNBURY"
    assert s["address"] == "302 Blair St"
    assert s["trading_name"] == "Vibe Bunbury South"


def test_parse_empty_feed_returns_empty() -> None:
    # BOM + no items.
    assert FuelWatchClient.parse("﻿<rss><channel></channel></rss>") == []


def test_parse_skips_item_without_price() -> None:
    feed = "﻿<rss><channel><item><brand>Vibe</brand><location>X</location></item></channel></rss>"
    assert FuelWatchClient.parse(feed) == []
