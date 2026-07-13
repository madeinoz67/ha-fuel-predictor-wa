# Architecture

## Data flow

```
        OFFLINE (dev machine)                      LIVE (Home Assistant)
 ┌────────────────────────────┐          ┌──────────────────────────────────┐
 │ data.wa.gov.au (24yr CSV)  │          │  FuelWatch API (today/tomorrow)  │
 │          │                 │          │            │                     │
 │   tools/download_history   │          │   fuelwatch.py (async aiohttp)   │
 │          ▼                  │          │            ▼                     │
 │   tools/train.py            │  seed    │   coordinator.py                 │
 │          │                  │ ───────► │   ├─ history.py (local append)   │
 │          ▼                  │ models/  │   ├─ predictor.py (numpy/pandas) │
 │   models/*.pkl              │          │   └─ sensor.py                   │
 └────────────────────────────┘          │            ▼                     │
                                          │   cheapest_day + cheapest_station │
                                          └──────────────────────────────────┘
```

## Two-tier history

- **Bulk seed** — 24+ years from [data.wa.gov.au](https://catalogue.data.wa.gov.au/dataset/fuelwatch-historic-fuel-prices),
  consumed offline by `tools/train.py` → bundled model artifacts.
- **Live append** — each poll appends today's prices to a local CSV
  (`history.py`), so the dataset keeps growing. Predictions never block on
  local-history depth because the seed already covers it.

## Forecast horizon

| Day | Source |
|-----|--------|
| 1 (today) | known — live FuelWatch |
| 2 (tomorrow) | known — live FuelWatch (after ~14:30 AWST) |
| 3–7 | forecast — seasonal model |

## Constraints

- Forecaster is **numpy/pandas only** (no sklearn/onnx) — keeps HA deps minimal.
- FuelWatch client is **vendored async** (`aiohttp`) — not the `fuelwatcher` PyPI dep.
- `iot_class: cloud_polling`.
- Config is **config-flow only** (no `configuration.yaml`).

## FuelWatch quirk

Tomorrow's price is only available after retailers lodge it (~14:30 AWST). The
coordinator polls twice daily and degrades gracefully (tomorrow → forecast)
before the cutoff.
