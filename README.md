# Fuel Predictor WA

A Home Assistant custom integration that predicts the **cheapest day to buy
fuel** in Western Australia over a configurable horizon (default 7 days), and
lists **today's cheapest nearby stations**. Built on the State's legislated
**FuelWatch** dataset.

> **Status: alpha — functional and live-validated.** 120 unit tests pass; the
> forecaster is a cycle-aware empirical-fade model (beats an average-baseline
> on walk-forward holdout) with a wholesale TGP leading-indicator drift term.
> Validated on a live Home Assistant install. Pre-1.0 — expect rough edges.

## Why

WA law ([Petroleum Products Pricing Act 1983](https://www.consumerprotection.wa.gov.au/fuelwatch-and-fuel-prices))
fixes each day's fuel price in advance — retailers lodge next-day prices by
~14:00, and there is no intraday movement. So:

- **Days 1–2** (today/tomorrow) are *known*.
- **Days 3+** are a genuine *forecast* — and the WA price cycle makes that
  forecast learnable.

The forecaster trains on **~24 months of history** (monthly FuelWatch CSVs from
the WA blob store, fetched automatically on first install), so predictions work
from day one — no weeks-long cold-start scrape.

## Sensors

| Sensor | Description |
|--------|-------------|
| `cheapest_day` | Cheapest day in the horizon + predicted c/L. Attributes: `cheapest_date`, `cheapest_source`, the full **`horizon`** list (see below), and live `cycle_pos` / `expected_next_hike_in_days`. |
| `cheapest_station_today` | Cheapest nearby station now (price/brand/suburb/address) + ranked list. |
| `Day 1`–`Day 14` | One entity per horizon day — its state is that day's predicted c/L, attributes `date` + `source` (`known`/`forecast`). Plot these directly for per-day history. |
| `model_fit` | Holdout goodness-of-fit. State = MAE (c/L); attributes: `model_kind`, `beats_baseline`, `mape_pct`, `improvement_pct`, `cycle_len_days`, `n_hikes`, `trained_at`. |
| `forecast_accuracy` | Forecast-vs-actual accuracy from the persisted ledger. State = overall MAE; attributes: `mae_by_days_out`, `bias`, `recent` pairs, coverage. |
| `Training status` | `untrained → training → ready` / `error` lifecycle. |

### The `horizon` attribute

`sensor.cheapest_day` exposes the full forward curve as a `horizon` attribute —
a list of one point per day, each carrying both a `power_predictor_24h`-style
`{time, value}` pair (for drop-in ApexCharts `data_generator` configs) and
richer fields:

```json
{
  "time": "2026-07-16T00:00:00+08:00",   // AWST midnight — fixed to Australia/Perth (WA has no DST)
  "value": 175.9,                         // = price_cpl
  "date": "2026-07-16",
  "ts": 1784160000000,                    // epoch-ms (UTC midnight)
  "price": 175.9,
  "source": "forecast"                    // "known" (today/tomorrow) or "forecast"
}
```

`time` is anchored to AWST (`Australia/Perth`, permanent UTC+8) rather than the
HA instance timezone, because FuelWatch prices are fixed per AWST day.

## Install (HACS)

1. In Home Assistant: **HACS → Integrations** → ⋮ (top-right) → **Custom repositories**.
2. Paste `https://github.com/madeinoz67/ha-fuel-predictor-wa`, set category to **Integration**, click **Add**.
3. Back in **Integrations**, find **Fuel Predictor WA** → **Download**.
4. **Restart Home Assistant** (Settings → System → ⋮ → Restart Home Assistant).
5. **Settings → Devices & Services → Add Integration** → search **Fuel Predictor WA**.
6. Pick your **fuel type**, **suburb**, **search radius**, and **forecast horizon**.
7. The **Training status** sensor goes `untrained → training → ready` on first run (downloads ~24 months of history in the background, ~30–60 s). The live today/tomorrow and cheapest-station sensors work immediately.

## Configuration

Config-flow only (no `configuration.yaml` keys). Defaults:

| Key | Default | Notes |
|-----|---------|-------|
| `product` | Unleaded 91 | ULP91 / Premium 95 / Premium 98 / Diesel / LPG / E85 |
| `suburb` | — | WA suburb (FuelWatch search anchor) |
| `surrounding` | true | Include surrounding suburbs |
| `radius_km` | 10 | Station search radius |
| `forecast_horizon_days` | 7 | Total horizon length (1–2 known, rest forecast; up to 14) |
| `station_limit` | 5 | How many cheap stations to list today |
| `min_stations` | 40 | Local catchment gate — expand the search until ≥ this many stations (accuracy sweet spot) |

## First-run training (automatic)

On first setup the integration downloads ~24 months of FuelWatch history
(deterministic monthly CSVs from the WA blob store), filters to your fuel type
and local catchment, and trains the forecaster in the background. A `Training
status` sensor shows `untrained → training → ready`. Live today/tomorrow
sensors work immediately. Call the `fuel_predictor_wa.retrain` service to
refresh on demand.

The model retrains periodically (every ~30 days) and refits daily at **15:00
AWST** — 30 minutes after FuelWatch publishes tomorrow's prices — so it always
trains on the freshest data.

## Charting — ApexCharts example

The `horizon` attribute is shaped for [apexcharts-card](https://github.com/RomRider/apexcharts-card).
This card plots **actual price history on the left**, the **forecast on the
right**, a **NOW line down the middle**, and a **±MAE fit band** (model
uncertainty, read live from `sensor.model_fit`) around the forecast:

```yaml
type: custom:apexcharts-card
header:
  show: true
  title: Fuel price — actual & forecast
  show_states: true
  colorize_states: true
graph_span: 20d              # 10d past + 10d future → now sits in the middle
span:
  start: day
  offset: "-10d"
all_series_config:
  type: line
  stroke_width: 2
  float_precision: 1
  curve: stepline
series:
  # Actual price history (left of now) — today's cheapest-station state history
  - entity: sensor.cheapest_station_today
    name: Actual
    color: "#03a9f4"
    extend_to: now
    group_by: { func: last, duration: 1d, fill: last }
  # Forecast (right of now) — the `horizon` attribute, dashed
  - entity: sensor.cheapest_day
    name: Forecast
    color: "#ff9800"
    stroke_dash: 6
    data_generator: |
      const h = entity.attributes.horizon || [];
      return h.map(p => [new Date(p.time).getTime(), p.value]);
  # ±MAE fit band — upper bound (model uncertainty from sensor.model_fit)
  - entity: sensor.cheapest_day
    name: Fit +MAE
    color: "#ff9800"
    opacity: 0.65
    stroke_dash: 2
    stroke_width: 1.5
    show: { in_legend: false }
    data_generator: |
      const h = entity.attributes.horizon || [];
      const mae = parseFloat(hass?.states?.['sensor.model_fit']?.state) || 0;
      return h.filter(p => p.source !== 'known')
              .map(p => [new Date(p.time).getTime(), p.value + mae]);
  # ±MAE fit band — lower bound
  - entity: sensor.cheapest_day
    name: Fit -MAE
    color: "#ff9800"
    opacity: 0.65
    stroke_dash: 2
    stroke_width: 1.5
    show: { in_legend: false }
    data_generator: |
      const h = entity.attributes.horizon || [];
      const mae = parseFloat(hass?.states?.['sensor.model_fit']?.state) || 0;
      return h.filter(p => p.source !== 'known')
              .map(p => [new Date(p.time).getTime(), p.value - mae]);
apex_config:
  annotations:
    xaxis:
      - x: "{{ (now().timestamp() | int) * 1000 }}"   # dynamic NOW line
        borderColor: "#9e9e9e"
        borderWidth: 2
        label:
          text: NOW
          borderColor: "#9e9e9e"
          style: { color: "#fff", background: "#9e9e9e" }
```

`data_generator` runs with `entity, start, end, hass, moment` in scope, so the
band reads `sensor.model_fit` live and widens/narrows as the MAE updates.

## Architecture

```
ON-INSTALL (HA background task)         OFFLINE (optional, dev machine)
 FuelWatch blob store (24 monthly CSVs)   data.wa.gov.au (24yr bulk CSV)
        │                                           │
   historic_client ──► trainer ──► model.pkl   tools/train.py ──► model.pkl
        │                                              (pre-seed alternative)
   catchment.py (local suburb set, min_stations gate)
        │
FuelWatch API (today/tmrw) ──► coordinator ──► predictor ──► sensors
                                  ▲                    │
                  history.py (daily append)      forecast_ledger (accuracy)
                  15:00 AWST refit
```

- **Forecaster**: cycle-aware empirical-fade curve anchored to the live known
  price (`numpy` + `pandas` only — no sklearn/onnx; the pure-fade path beats
  the prior gradient-boosting model and installs cleanly on HA's Python 3.14).
  A wholesale TGP leading-indicator drift term adjusts the level.
- **Client**: vendored async FuelWatch client (`aiohttp`), polled twice daily,
  plus a post-publication fetch + refit at 15:00 AWST.
- **Catchment**: resolves a local suburb set around the anchor, gated by
  `min_stations` so the model trains on enough nearby price signal.
- **Accuracy ledger**: each issued forecast is recorded and later paired with
  the actual — drives `forecast_accuracy` and `mae_by_days_out`.
- **`iot_class`**: `cloud_polling`.

## Development

```bash
pip install -r requirements-test.txt
ruff check .
mypy custom_components
pytest
```

Training runs automatically on first install and can be re-run anytime via the
`fuel_predictor_wa.retrain` service — no manual steps needed. The offline
`tools/` scripts (`download_history.py`, `train.py`, `backtest.py`) are a
secondary path for pre-seeding a model and running accuracy backtests.

## License

MIT — see [LICENSE](LICENSE).
