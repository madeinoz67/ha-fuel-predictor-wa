# Architecture

## Data flow

```
ON-INSTALL (config flow + setup_entry)        OFFLINE (dev machine; pre-seed only)
  geocode.py → anchor suburb (HA lat/lon)     tools/download_history.py (data.wa.gov.au)
  load model.pkl if present                   tools/train.py → model.pkl
  first refresh → cold-start train if unfitted
  schedule 15:00 AWST refit
  register fuel_predictor_wa.retrain

RUNTIME — coordinator poll loop (twice daily)
  FuelWatch API (today/tmrw) ── known{today,tomorrow} ──┐
                                                          ▼
  coordinator ──► predictor.predict ──► sensors
    │            (pure-fade + TGP drift)   (cheapest_day, cheapest_station_today,
    │                                      Day 1–14, model_fit, forecast_accuracy)
    └─► forecast_ledger (snapshot + actual → accuracy, each poll)

  Training is fire-and-forget — predict never waits. Three triggers → _async_train_background:
    15:00 AWST schedule · 24h _should_refit gate · fuel_predictor_wa.retrain service

  Azure Blob (FuelWatch): 24mo retail CSVs + 3yr wholesale TGP ─► historic_client ─┐
  catchment.py: suburb set, min_stations gate (lazy, disk-cached) ─────────────────┤
                                                                                    ▼
                                          trainer.assemble_and_train ──► model.pkl
```

## Two-tier history

- **On-install seed (default)** — ~24 monthly retail CSVs streamed from the WA
  Azure blob store (`historic_client.async_fetch_month_cached`), filtered to the
  configured product and local catchment, fit on the HA host by
  `trainer.assemble_and_train` inside a coordinator background task. No dev
  machine, no manual download. Months are disk-cached (`bulk_cache/`); the
  current + previous month re-download because they still gain days.
- **Bulk seed (optional, offline)** — 24+ years from
  [data.wa.gov.au](https://catalogue.data.wa.gov.au/dataset/fuelwatch-historic-fuel-prices),
  consumed offline by `tools/download_history.py` → `tools/train.py` → a bundled
  `model.pkl`. Use this to pre-seed in CI or ship a pinned artifact.
- **No live append CSV.** The runtime does **not** append to a local price CSV.
  The dataset refreshes by re-fetching the rolling monthly blob CSVs on each
  refit. (A `history.py` helper exists for the offline `tools/train.py` path
  only; it is not in the runtime loop.) Forecast snapshots + actuals are
  recorded in the `forecast_ledger`, not a price-history CSV.

## On-install / refit training path

```
_async_train_background (fire-and-forget; first poll if unfitted, else on a trigger)
   │
   ├── _async_resolve_catchment          # lazy: disk cache → else FuelWatch suburbs API
   │       └── resolve_catchment (min_stations gate) → cached catchment.json
   ├── _async_fetch_tgp                  # 3yr wholesale TGP CSVs (Azure blob)
   ├── historic_client.async_fetch_month_cached  # 24mo retail CSVs (Azure blob, cached)
   │       └── parse_csv (executor, product-filtered, catchment-filtered at collapse)
   ├── trainer.assemble_and_train        # series + tgp_series → fit → predictor
   │       └── save_model → model.pkl (versioned, MODEL_VERSION guard)
   └── status: training → ready  (or → error on <3 months)
```

The blob endpoint serves month-sized CSVs (~9 MB each); `parse_csv` filters to
the product during iteration so resident memory stays bounded. On failure the
lifecycle moves to `error` and the live sensors continue to degrade gracefully
(tomorrow → forecast).

## Retrain triggers

All three route to `_async_train_background`:

| Trigger | When |
|---------|------|
| 15:00 AWST schedule | Wall-clock `async_track_time_change` — shortly after FuelWatch publishes tomorrow's prices (~14:30). |
| 24h staleness gate | `_should_refit` — checked every poll; fires if the model is >24h old (fallback if HA was down at 15:00). |
| `fuel_predictor_wa.retrain` service | Manual, on demand. |

So the model effectively refits **daily**. (A `RETRAIN_INTERVAL_DAYS=30`
constant existed historically but was never wired in and has been removed.)

## Training status lifecycle

| State | Meaning |
|-------|---------|
| `untrained` | No artifact yet; on-install task not started or finished |
| `training` | Background download + fit in progress |
| `ready` | Artifact loaded; predictor serving forecasts |
| `error` | Training failed (<3 months of data, network, etc.); live sensors still run |

## Forecast horizon

Configurable (`forecast_horizon_days`, default 7, up to 14). Each day:

| Day | Source |
|-----|--------|
| 1 (today) | known — live FuelWatch |
| 2 (tomorrow) | known — live FuelWatch (after ~14:30 AWST) |
| 3+ | forecast — cycle-aware empirical-fade + wholesale TGP drift |

## Sensors

| Sensor | Description |
|--------|-------------|
| `cheapest_day` | Cheapest day in the horizon + predicted c/L. Attributes: `cheapest_date`, `cheapest_source`, the full **`horizon`** list, live `cycle_pos` / `expected_next_hike_in_days`. |
| `cheapest_station_today` | Cheapest nearby station now + ranked list. |
| `Day 1`–`Day 14` | One entity per horizon day (state = predicted c/L; attrs `date`, `source`). |
| `model_fit` | Holdout MAE + `model_kind`, `beats_baseline`, `mape_pct`, `cycle_len_days`, `n_hikes`, `trained_at`. |
| `forecast_accuracy` | Ledger MAE + `mae_by_days_out`, `bias`, `recent`, coverage. |
| `Training status` | `untrained → training → ready` / `error`. |

The `horizon` attribute on `cheapest_day` is a list of `{time, value, date, ts, price, source}` per day — shaped for an apexcharts-card `data_generator` (see the README for a worked example).

## Constraints

- Forecaster is **numpy/pandas only** (no sklearn/onnx) — the pure-fade path
  beats the prior gradient-boosting model and installs cleanly on HA's Python 3.14.
- A **wholesale TGP leading-indicator drift term** shifts the forecast level;
  degrades gracefully (no drift) if the TGP fetch fails.
- FuelWatch client is **vendored async** (`aiohttp`) — not the `fuelwatcher` PyPI dep.
- `iot_class: cloud_polling`.
- Config is **config-flow only** (no `configuration.yaml`).

## FuelWatch quirk

Tomorrow's price is only available after retailers lodge it (~14:30 AWST). The
coordinator polls twice daily and degrades gracefully (tomorrow → forecast)
before the cutoff. The 15:00 AWST refit fires shortly after publication so the
model trains on the freshest data.
