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
FUELWATCH_ENDPOINT = "https://www.fuelwatch.wa.gov.au/fuelwatch/fuelWatchRSS"
FUELWATCH_SUBURBS_ENDPOINT = "https://www.fuelwatch.wa.gov.au/api/sites/suburbs"
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
CONF_MIN_STATIONS = "min_stations"  # local catchment gate: expand to >= this many stations

# --- Defaults -------------------------------------------------------------
DEFAULT_PRODUCT = PRODUCT_UNLEADED
DEFAULT_SURROUNDING = True
DEFAULT_RADIUS_KM = 10
DEFAULT_FORECAST_HORIZON_DAYS = 7
DEFAULT_STATION_LIMIT = 5
DEFAULT_MIN_STATIONS = 40  # catchment gate: ~regional sweet spot (accuracy spike)
MIN_MIN_STATIONS = 5
MAX_MIN_STATIONS = 200

# FuelWatch publishes tomorrow's price after ~14:30 AWST; refresh twice daily.
UPDATE_INTERVAL_MINUTES = 720

# Local daily-append history (under HA config dir) — grows the dataset over
# time, but predictions work from day one via the bulk CSV seed.
HISTORY_FILENAME = "fuel_predictor_wa_history.csv"

# FuelWatch reports prices in cents per litre.
UNIT_CENTS_PER_LITRE = "c/L"

# --- Historical CSV source (Azure Blob; deterministic, not filterable) ---
HISTORIC_CSV_BASE = "https://warsydprdstafuelwatch.blob.core.windows.net/historical-reports"
HISTORIC_CSV_TEMPLATE = "FuelWatchRetail-{mm:02d}-{yyyy}.csv"

# Product code -> FuelWatch CSV PRODUCT_DESCRIPTION string.
# Verified against a real monthly CSV: ULP, PULP, 98 RON, Diesel, LPG, E85.
PRODUCT_CSV_DESCRIPTION: dict[int, str] = {
    PRODUCT_UNLEADED: "ULP",
    PRODUCT_P95: "PULP",
    PRODUCT_P98: "98 RON",
    PRODUCT_DIESEL: "Diesel",
    PRODUCT_LPG: "LPG",
    PRODUCT_E85: "E85",
}

# --- On-install training tuning ---
HISTORY_MONTHS_TARGET = 24
MIN_MONTHS_TO_TRAIN = 3
MIN_MONTHS_FULL_MODEL = 6  # gates the full HGBR path (~3 cycles min for stable cycle detection)
RETRAIN_INTERVAL_DAYS = 30
RETRAIN_INTERVAL_HOURS = 24  # periodic background refit cadence (after a successful poll)
TGP_LAG_DAYS = 10  # wholesale (Singapore-Mogas-derived) TGP → retail pass-through window

# --- Storage (under HA config dir: <config>/fuel_predictor_wa/<entry_id>/) ---
STORAGE_DIRNAME = "fuel_predictor_wa"
MODEL_FILENAME = "model.pkl"
HISTORY_SUBDIR = "history"
BULK_CACHE_DIRNAME = "bulk_cache"  # cached monthly historical CSVs (immutable months)
CATCHMENT_FILENAME = "catchment.json"  # resolved local catchment (suburb set + meta)

# --- Global leading-indicator prices (Yahoo Finance chart API) ---
# RBOB gasoline (~$/gallon), Brent crude (~$/barrel), AUD/USD FX.
# Global prices lead WA retail by ~1-2 weeks (pass-through), so they feed the
# predictor's global-feature layer at a lag.
YAHOO_BASE_URL = "https://query1.finance.yahoo.com/v8/finance/chart"
YAHOO_SYMBOLS = ("RB=F", "BZ=F", "AUDUSD=X")  # RBOB gasoline, Brent crude, AUD/USD
GLOBAL_FEATURE_LAG_DAYS = 7  # global prices lead retail by ~1-2 weeks (pass-through)

# --- Status states (status diagnostic sensor) ---
STATUS_UNTRAINED = "untrained"
STATUS_TRAINING = "training"
STATUS_RETRAINING = "retraining"
STATUS_READY = "ready"
STATUS_ERROR = "error"
