# CLAUDE.md — Fuel Predictor WA

Home Assistant custom integration (HACS). Python 3.12+. Predicts the cheapest
day to buy fuel in WA over 7 days using the FuelWatch dataset.

## Domain knowledge (read first)

- **FuelWatch scheme**: WA retailers lodge next-day prices by ~14:00 AWST under
  the Petroleum Products Pricing Act 1983. Prices are **fixed per day — no
  intraday movement**. The public API serves **today + tomorrow only**.
- Therefore days 1–2 of any horizon are *known*; days 3+ are *forecast*.
- **Cold start is solved**: 24+ years of daily history is published on
  [data.wa.gov.au](https://catalogue.data.wa.gov.au/dataset/fuelwatch-historic-fuel-prices)
  (resource id in `const.py`). The trainer downloads this; no scrape-and-wait.
- FuelWatch reports prices in **cents per litre (c/L)**.

## Architecture rules

- **Forecaster must stay numpy/pandas only** (no sklearn/onnx/torch) — keeps the
  HA `requirements` minimal, mirroring `isaacjmannion/ha-power-predictor`.
- **Live API client is vendored** in `fuelwatch.py` (async, `aiohttp`). Do NOT
  depend on the `fuelwatcher` PyPI package — it's thin, sync, and unmaintained.
- **History is two-tier**: bulk CSV seeds training (offline, `tools/`); the live
  integration appends each day's poll to a local CSV (`history.py`) so the
  dataset grows, but predictions never block on local history depth.
- **`iot_class` is `cloud_polling`** (FuelWatch is a remote government API).
- **Config is config-flow only** — no `configuration.yaml` keys (HA convention
  for modern integrations; the HA best-practice skill flags YAML as wrong here).

## Layout

```
custom_components/fuel_predictor_wa/   the integration (only this dir ships to HA)
  __init__.py    setup entry, creates coordinator
  coordinator.py DataUpdateCoordinator: fetch live + run predictor
  fuelwatch.py   async FuelWatch client (today/tomorrow XML)
  history.py     bulk-CSV load + daily append + windowing
  predictor.py   numpy/pandas 7-day forecaster
  sensor.py      cheapest-day + cheapest-station-today sensors
  config_flow.py product / suburb / radius / horizon
tools/           OFFLINE — download_history.py, train.py (not shipped to HA)
tests/           pytest + pytest-homeassistant-custom-component
```

## Testing

`pytest` (asyncio auto mode). Integration tests use
`pytest-homeassistant-custom-component` harness. Pure functions (`predictor`,
`history` windowing) get unit tests first — the forecaster is the highest-risk
logic and should be property-tested where feasible.

## Conventions

- Ruff (E/F/I/UP/B/SIM/C4) + mypy; line length 100; `from __future__ import
  annotations` in every module.
- Commit messages: Conventional Commits (`feat:`, `fix:`, `chore:`…);
  `cliff.toml` generates the changelog.
- Keep `manifest.json` `version` in sync on releases.

## Out of scope (for now)

- Intrastate region search beyond suburb+surrounding.
- Non-WA states (NSW FuelCheck etc.) — the engine is pluggable but only
  FuelWatch is wired.
- Real-time price alerts (the scheme has no intraday data to alert on).
