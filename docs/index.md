# Fuel Predictor WA

A Home Assistant integration that predicts the **cheapest day to buy fuel** in
Western Australia over the next 7 days, and lists **today's cheapest nearby
stations** — built on the State's legislated [FuelWatch](https://www.fuelwatch.wa.gov.au/)
dataset.

## How it works

WA law fixes each day's fuel price in advance (next-day prices lodged by ~14:00
AWST; no intraday movement). So the first two days of any horizon are *known*,
and only days 3–7 need forecasting. The WA price cycle is strongly seasonal, so
a lightweight seasonal model trained on the government's 24-year history works
well — predictions are useful from day one, with no cold-start scrape.

See [Architecture](architecture.md) for the data flow and module layout.

## Quick start

1. Add this repo in HACS (category: Integration), install, restart HA.
2. **Settings → Devices & Services → Add Integration → Fuel Predictor WA**.
3. Pick your fuel type, suburb, and search radius.

## Train the model (optional but recommended)

```bash
python tools/download_history.py
python tools/train.py
```
