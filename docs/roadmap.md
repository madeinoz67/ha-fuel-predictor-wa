# Roadmap / Follow-ups

The production forecaster is a **cycle-aware empirical-fade model anchored to the
live known price, with a wholesale TGP leading-indicator drift term** (numpy
only). It beats an average-baseline on walk-forward holdout — see
`sensor.model_fit` (`beats_baseline`, MAE) and `sensor.forecast_accuracy`
(`mae_by_days_out`) for live numbers.

> The earlier HistGradientBoostingRegressor + offset-calibration path was
> **removed** — ML-6 found the pure-fade forecast beats it, and the ML
> dependency doesn't install on HA's Python 3.14. The backtest figures
> previously quoted here (MAE 5.5 c/L, post-hike 7.8, hit-rate 47%) were from
> that removed model and are superseded; read current accuracy from the sensors.

These are the deferred enhancements, in rough priority order.

## Done

- **Daily auto-refit.** The model refits daily — a 15:00 AWST wall-clock
  schedule (after FuelWatch publishes tomorrow's prices) plus a 24h staleness
  gate — and is retrainable on demand via `fuel_predictor_wa.retrain`. (The
  previously-deferred "30-day timer" is obsolete; `RETRAIN_INTERVAL_DAYS` was
  never wired in and has been removed.)
- **Wholesale TGP leading-indicator drift.** A β-fit wholesale Terminal Gate
  Price return over a 10-day lag shifts the forecast level (v0.4.0). Captures
  pass-through the retail-only cycle can't see; degrades gracefully if the TGP
  fetch fails.
- **Forecast accuracy ledger.** Each issued forecast is snapshotted and later
  paired with the actual — drives `sensor.forecast_accuracy` + `mae_by_days_out`.
- **Per-day forecast entities (Day 1–14)** + the `horizon` attribute
  (`time`/`value` for apexcharts).
- **Dead-code cleanup.** Removed unused consts (`MIN_MONTHS_FULL_MODEL`,
  `HISTORY_FILENAME`) and the dormant Yahoo `global_features.py` + its test +
  spike (superseded by the TGP drift). `history.py` kept as an offline
  `tools/train.py` helper.
- **`cheapest()` guard.** An empty horizon now raises `ValueError` (not a
  cryptic `IndexError`); the type-safe key also cleared the predictor mypy error.
- **Backtest acceptance gate.** Gates on the rolling-origin MAE / beats-baseline
  (the real ~120-day test), not the in-fit metrics that go n/a.
- **Predict-path comments.** Documented the clamp-bound derivation.

## Accuracy

- **Live-cycle-amplitude blend.** The empirical fade curve is a static
  historical average; blending in the most-recent cycle's observed amplitude
  should sharpen the fade and push cheapest-day accuracy higher.
- **Hike-hazard model.** A lightweight classifier for next-hike
  timing/probability — the cycle length varies and next-hike timing is partly
  exogenous, which caps single-point cheapest-day accuracy. Would give honest
  uncertainty per forecast day.
- **Per-station forecasting.** Currently per-product daily-min; per-station
  would enable "cheapest station N days ahead."

## Operational

- Cheapest-day is an *advisor* (not >50% hit-rate) — the sensor attributes carry
  the horizon + sources + MAE band so the user can judge confidence.
