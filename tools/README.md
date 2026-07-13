# tools/ — offline model pipeline

These scripts run **offline** (on your dev machine, not inside Home Assistant).
They produce the trained artifacts the live integration loads.

## Pipeline

```bash
# 1. Pull ~24 years of FuelWatch history from data.wa.gov.au
python tools/download_history.py --out data/fuelwatch_history.csv

#    Verify the CSV header columns match the detection lists in
#    custom_components/fuel_predictor_wa/history.py; extend them if needed.

# 2. Fit the baseline forecaster per fuel product -> models/
python tools/train.py --history data/fuelwatch_history.csv
#    or one product: --product 1   (Unleaded 91)
```

## What each does

| Script | Purpose |
|--------|---------|
| `download_history.py` | CKAN `package_show` → resolves the CSV resource URL → streams to `data/` |
| `train.py` | Loads the CSV, fits `FuelPricePredictor` per product, pickles params to `models/` |

## Notes

- The baseline is seasonal (weekday mean + recent level). Replace the predictor
  internals with a stronger cycle model when ready — the `fit`/`predict` contract
  stays stable.
- Trained artifacts (`*.pkl`) are gitignored; regenerate after history updates.
- FuelWatch product codes are in `custom_components/fuel_predictor_wa/const.py`.
