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

- **On-install seed (default)** — ~24 monthly CSVs streamed from the WA blob
  store (`historic_client.month_url`), filtered to the configured product in
  `historic_client.parse_csv`, fit on the HA host by `trainer.assemble_and_train`
  inside a coordinator background task. No dev machine, no manual download.
- **Bulk seed (optional)** — 24+ years from [data.wa.gov.au](https://catalogue.data.wa.gov.au/dataset/fuelwatch-historic-fuel-prices),
  consumed offline by `tools/train.py` → bundled model artifacts. Use this if
  you want to pre-seed the model in CI or ship a pinned artifact.
- **Live append** — each poll appends today's prices to a local CSV
  (`history.py`), so the dataset keeps growing. Predictions never block on
  local-history depth because a seed (on-install or bulk) already covers it.

## On-install data path

```
 first refresh (coordinator)
   │
   ├── async_create_task(_async_train_background)   # non-blocking
   │       │
   │       ├── historic_client.fetch_month          # async blob GET
   │       │       └── parse_csv (executor)         # streams + product-filters
   │       ├── trainer.assemble_and_train           # series → fit → pickle
   │       │       └── writes models/*.pkl
   │       └── status: training → ready  (or → error on <3 months)
   │
   └── _async_update_data                           # live today/tomorrow, runs now
```

The blob endpoint serves month-sized CSVs (~9 MB each); `parse_csv` filters
to the product during iteration so resident memory stays bounded. On failure
the lifecycle moves to `error` and the live sensors continue to degrade
graceously (tomorrow → forecast).

## Training status lifecycle

| State | Meaning |
|-------|---------|
| `untrained` | No artifact yet; on-install task not started or finished |
| `training` | Background download + fit in progress |
| `ready` | Artifact loaded; predictor serving forecasts |
| `error` | Training failed (<3 months of data, network, etc.); live sensors still run |

`fuel_predictor_wa.retrain` re-runs the background path on demand (`ready` →
`training` → `ready`/`error`).

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
