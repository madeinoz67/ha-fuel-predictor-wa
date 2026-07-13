# Fuel Predictor WA

A Home Assistant custom integration that predicts the **cheapest day to buy
fuel** in Western Australia over the next 7 days, and lists **today's cheapest
nearby stations**. Built on the State's legislated **FuelWatch** dataset.

> ⚠️ **Status: early alpha (scaffold)** — functional and unit-tested (22 tests), but
> **not yet validated on a live Home Assistant install**. The forecaster is a
> baseline seasonal model (expect to tune it), and the live FuelWatch
> today/tomorrow endpoint is a best-effort default pending a real-request check.

## Why

WA law ([Petroleum Products Pricing Act 1983](https://www.consumerprotection.wa.gov.au/fuelwatch-and-fuel-prices))
fixes each day's fuel price in advance — retailers lodge next-day prices by
~14:00, and there is no intraday movement. So:

- **Days 1–2** (today/tomorrow) are *known*.
- **Days 3–7** are a genuine *forecast* — and the WA price cycle makes that
  forecast learnable.

The forecaster trains on **~24 months of history** (monthly FuelWatch CSVs,
fetched automatically on first install), so predictions work from day one —
no weeks-long cold-start scrape.

## Sensors

| Sensor | Description |
|--------|-------------|
| `cheapest_day` | Cheapest day in the 7-day horizon + predicted c/L; full 7-day forecast in attributes |
| `cheapest_station_today` | Cheapest nearby station now (price/brand/suburb/address) + ranked list |
| `Training status` | `untrained → training → ready` / `error` lifecycle |

## Install (HACS)

1. In Home Assistant: **HACS → Integrations** → ⋮ (top-right) → **Custom repositories**.
2. Paste `https://github.com/madeinoz67/ha-fuel-predictor-wa`, set category to **Integration**, click **Add**.
3. Back in **Integrations**, find **Fuel Predictor WA** → **Download**.
4. **Restart Home Assistant** (Settings → System → ⋮ → Restart Home Assistant).
5. **Settings → Devices & Services → Add Integration** → search **Fuel Predictor WA**.
6. Pick your **fuel type**, **suburb**, and **search radius**.
7. The **Training status** sensor goes `untrained → training → ready` on first run (downloads ~24 months of history in the background, ~30–60 s). The live today/tomorrow and cheapest-station sensors work immediately.

## Configuration

| Key | Default | Notes |
|-----|---------|-------|
| `product` | Unleaded 91 | ULP91 / 95 / 98 / Diesel / LPG / E85 |
| `suburb` | — | WA suburb (FuelWatch search anchor) |
| `surrounding` | true | Include surrounding suburbs |
| `forecast_horizon_days` | 7 | Days 3–7 forecast, 1–2 known |
| `station_limit` | 5 | How many cheap stations to list today |

## First-run training (automatic)

On first setup the integration downloads ~24 months of FuelWatch history
(deterministic monthly CSVs from the WA blob store), filters to your fuel type,
and trains the forecaster in the background. A `Training status` sensor shows
`untrained → training → ready`. Live today/tomorrow sensors work immediately.
Call the `fuel_predictor_wa.retrain` service to refresh on demand.

## Architecture

```
ON-INSTALL (HA background task)         OFFLINE (optional, dev machine)
 FuelWatch blob store (24 monthly CSVs)   data.wa.gov.au (24yr bulk CSV)
        │                                           │
   historic_client ──► trainer ──► models/    tools/train.py ──► models/
                                          (pre-seed alternative)
FuelWatch API (today/tmrw) ──► coordinator ──► predictor ──► sensors
                                  ▲
                  history.py (daily append → grows local dataset)
```

- **Forecaster**: `numpy` + `pandas` only (no sklearn/onnx).
- **Client**: vendored async FuelWatch client (`aiohttp`), polled twice daily.
- **Training status**: `untrained → training → ready` (or `error`). Retraining
  is triggered by the `fuel_predictor_wa.retrain` service.
- **`iot_class`**: `cloud_polling`.

## Development

```bash
pip install -r requirements-test.txt
ruff check .
mypy custom_components
pytest
```

Training runs automatically on first install (see above) and can be re-run
anytime via the `fuel_predictor_wa.retrain` service — no manual steps needed.

The offline `tools/` scripts are a secondary path for pre-seeding a model. Note
`tools/download_history.py` targets the data.wa.gov.au CKAN entry (which is the
historic *page*, not a bulk CSV) and may need adjustment — the on-install blob
fetcher is the canonical training source.

## License

MIT — see [LICENSE](LICENSE).
