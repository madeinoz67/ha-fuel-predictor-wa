# Roadmap / Follow-ups

The cycle-aware ML model (HistGradientBoostingRegressor + offset calibration + empirical fade) is shipped and backtest-validated on real WA data: **MAE 5.5 c/L** (−73% vs the average baseline), **post-hike MAE 7.8 c/L**, **cheapest-day hit-rate 47%** (p≈0.003 vs 14% random). These are the deferred enhancements, in rough priority order.

## High value

- **Periodic auto-retrain (monthly).** Training currently runs once on install (+ on artifact/sklearn-version invalidation, + via the manual `fuel_predictor_wa.retrain` service). Wire the deferred 30-day timer so the model refreshes as new daily history accumulates and the cycle regime drifts.
- **Global leading-indicator features.** Lagged Singapore MOGAS / RBOB (`RB=F`), Brent (`BZ=F`), AUD/USD (`AUDUSD=X`) — self-fetched from Yahoo Finance (approved). They lead retail by ~1–2 weeks (pass-through) and capture regime shifts the retail-only cycle model can't see. Add as features in the regressor; degrade gracefully if Yahoo is unreachable.

## Accuracy

- **Live-cycle-amplitude blend.** The empirical fade curve is a static historical average; blending in the most-recent cycle's observed amplitude should sharpen the fade and push cheapest-day accuracy above the current 47%.
- **Hike-hazard model.** A lightweight classifier for next-hike timing/probability (the design's documented v2) — the cycle length varies and next-hike timing is partly exogenous, which caps single-point cheapest-day accuracy. Would give honest uncertainty per forecast day.
- **Per-station forecasting.** Currently per-product daily-min; per-station would enable "cheapest station N days ahead."

## Cleanup / hardening (low risk)

- **Backtest gate reporting** — `tools/backtest.py` prints `VERDICT: FAIL` because it reads the in-fit `post_hike_mae` (which is `n/a` when the stepped holdout samples no post-hike days). Point the printed gate at the rolling-origin `post_hike_mae` (the real, passing number).
- **`MIN_MONTHS_FULL_MODEL`** in `const.py` is dead code (the HGBR tier is actually gated by day-count + hike-count in the predictor) — wire it in or drop it.
- **`cheapest()`** raises on an empty `points` list — defensive `None` guard (coordinator always passes a non-empty horizon, so non-blocking).
- **Predict-path comments** — document the `elapsed` clamp and the clamp-bound derivation.

## Operational

- 30-day auto-retrain timer (see above) also covers the "model goes stale" case without a manual service call.
- Cheapest-day is an *advisor* (47%, not >50%) — the sensor attributes carry the horizon + sources so the user can judge confidence.
