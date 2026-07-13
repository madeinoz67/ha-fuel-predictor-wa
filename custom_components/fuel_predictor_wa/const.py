"""Constants for the Fuel Predictor WA integration.

Western Australia's FuelWatch scheme: retailers must lodge next-day prices by
~14:00 AWST under the Petroleum Products Pricing Act 1983. Prices are fixed for
the day — no intraday movement. The public API therefore only ever serves
``today`` and ``tomorrow``; longer horizons must be forecast.
"""
from __future__ import annotations

# --- Integration identity -------------------------------------------------
DOMAIN = "fuel_predictor_wa"
DOMAIN_TITLE = "Fuel Predictor WA"

# --- Live FuelWatch endpoint (today/tomorrow only) ------------------------
# Public XML service. Exact path to confirm during implementation; the service
# exposes the same data as the RSS feed at https://www.fuelwatch.wa.gov.au/tools/rss
FUELWATCH_ENDPOINT = "https://www.fuelwatch.wa.gov.au/retail/fuel/fuelwatch"
FUELWATCH_DAY_TODAY = "today"
FUELWATCH_DAY_TOMORROW = "tomorrow"

# --- Bulk historical source (cold-start training data) --------------------
# WA open-data (CKAN). Daily per-station prices since Jan 2001.
# https://catalogue.data.wa.gov.au/dataset/fuelwatch-historic-fuel-prices
HISTORY_CKAN_BASE = "https://catalogue.data.wa.gov.au"
HISTORY_DATASET_ID = "fuelwatch-historic-fuel-prices"
HISTORY_RESOURCE_ID = "903a3bfb-b8ea-4cff-8c24-47a05d40b112"

# --- FuelWatch product codes ----------------------------------------------
PRODUCT_UNLEADED = 1
PRODUCT_P95 = 2
PRODUCT_P98 = 4
PRODUCT_DIESEL = 5
PRODUCT_LPG = 6
PRODUCT_E85 = 11

PRODUCTS: dict[int, str] = {
    PRODUCT_UNLEADED: "Unleaded 91",
    PRODUCT_P95: "Premium 95",
    PRODUCT_P98: "Premium 98",
    PRODUCT_DIESEL: "Diesel",
    PRODUCT_LPG: "LPG",
    PRODUCT_E85: "E85",
}

# --- Config flow keys -----------------------------------------------------
CONF_PRODUCT = "product"
CONF_SUBURB = "suburb"
CONF_SURROUNDING = "surrounding"
CONF_RADIUS_KM = "radius_km"
CONF_FORECAST_HORIZON_DAYS = "forecast_horizon_days"
CONF_STATION_LIMIT = "station_limit"

# --- Defaults -------------------------------------------------------------
DEFAULT_PRODUCT = PRODUCT_UNLEADED
DEFAULT_SURROUNDING = True
DEFAULT_RADIUS_KM = 10
DEFAULT_FORECAST_HORIZON_DAYS = 7
DEFAULT_STATION_LIMIT = 5

# FuelWatch publishes tomorrow's price after ~14:30 AWST; refresh twice daily.
UPDATE_INTERVAL_MINUTES = 720

# Local daily-append history (under HA config dir) — grows the dataset over
# time, but predictions work from day one via the bulk CSV seed.
HISTORY_FILENAME = "fuel_predictor_wa_history.csv"

# FuelWatch reports prices in cents per litre.
UNIT_CENTS_PER_LITRE = "c/L"
