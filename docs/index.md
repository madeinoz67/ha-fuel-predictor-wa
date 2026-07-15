# Fuel Predictor WA

A Home Assistant integration that predicts the **cheapest day to buy fuel** in
Western Australia over a configurable horizon (default 7 days, up to 14), and
lists **today's cheapest nearby stations** — built on the State's legislated
[FuelWatch](https://www.fuelwatch.wa.gov.au/) dataset.

## How it works

WA law fixes each day's fuel price in advance (next-day prices lodged by ~14:00
AWST; no intraday movement). So the first two days of any horizon are *known*,
and only day 3 onward needs forecasting. The WA price cycle is learnable, so a
**cycle-aware empirical-fade model** — anchored to the live known price, with a
wholesale TGP leading-indicator drift term — trained on ~24 months of history
works well. Predictions are useful from day one, with no cold-start scrape.

See [Architecture](architecture.md) for the data flow and module layout, and the
[README](../README.md#sensors) for the full sensor list and an ApexCharts example.

## Quick start

1. Add this repo in HACS (category: Integration), install, restart HA.
2. **Settings → Devices & Services → Add Integration → Fuel Predictor WA**.
3. Pick your **fuel type**, **suburb**, **search radius**, **forecast horizon**,
   and **min stations** (local catchment gate).

The **Training status** sensor goes `untrained → training → ready` on first run
(downloads ~24 months of history in the background). The live today/tomorrow and
cheapest-station sensors work immediately.

## Train the model (optional, offline pre-seed)

On-install auto-training is the default and needs no dev machine. For a pinned
CI artifact or offline pre-seed:

```bash
python tools/download_history.py
python tools/train.py
```
