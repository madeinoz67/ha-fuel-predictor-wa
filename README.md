# Fuel Predictor WA

A Home Assistant custom integration that predicts the **cheapest day to buy
fuel** in Western Australia over the next 7 days, and lists **today's cheapest
nearby stations**. Built on the State's legislated **FuelWatch** dataset.

> Status: scaffold — structure complete, ML modules stubbed for implementation.

## Why

WA law ([Petroleum Products Pricing Act 1983](https://www.consumerprotection.wa.gov.au/fuelwatch-and-fuel-prices))
fixes each day's fuel price in advance — retailers lodge next-day prices by
~14:00, and there is no intraday movement. So:

- **Days 1–2** (today/tomorrow) are *known*.
- **Days 3–7** are a genuine *forecast* — and the WA price cycle makes that
  forecast learnable.

The forecaster trains on **24+ years of history** published on the WA open-data
portal ([FuelWatch Historic Fuel Prices](https://catalogue.data.wa.gov.au/dataset/fuelwatch-historic-fuel-prices)),
so predictions work from day one — no weeks-long cold-start scrape.

## Sensors

| Sensor | Description |
|--------|-------------|
| `cheapest_day` | Cheapest day in the 7-day horizon + predicted c/L |
| `forecast` | Predicted c/L for each of the next 7 days (attributes) |
| `cheapest_station_today` | Cheapest nearby station now (price/brand/suburb/address) |

## Install (HACS)

1. Add this repo as a custom repository in HACS (category: Integration).
2. Install, restart Home Assistant.
3. **Settings → Devices & Services → Add Integration → Fuel Predictor WA**.
4. Choose your **suburb**, **fuel type**, and **search radius**.

## Configuration

| Key | Default | Notes |
|-----|---------|-------|
| `product` | Unleaded 91 | ULP91 / 95 / 98 / Diesel / LPG / E85 |
| `suburb` | — | WA suburb (FuelWatch search anchor) |
| `surrounding` | true | Include surrounding suburbs |
| `forecast_horizon_days` | 7 | Days 3–7 forecast, 1–2 known |
| `station_limit` | 5 | How many cheap stations to list today |

## Architecture

```
data.wa.gov.au (24yr CSV) ──► tools/train.py ──► models/   (offline, seeds forecaster)
FuelWatch API (today/tmrw) ──► coordinator ──► predictor ──► sensors
                                  ▲
                  history.py (daily append → grows local dataset)
```

- **Forecaster**: `numpy` + `pandas` only (no sklearn/onnx) — mirrors
  [ha-power-predictor](https://github.com/isaacjmannion/ha-power-predictor).
- **Client**: vendored async FuelWatch client (`aiohttp`), polled twice daily.
- **`iot_class`**: `cloud_polling`.

## Development

```bash
pip install -r requirements-test.txt
ruff check .
mypy custom_components
pytest
```

Re-train the model after updating history:

```bash
python tools/download_history.py   # pull bulk CSV from data.wa.gov.au
python tools/train.py              # fit per-product forecasters → models/
```

## License

MIT — see [LICENSE](LICENSE).
