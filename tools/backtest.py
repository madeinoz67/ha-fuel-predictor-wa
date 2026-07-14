#!/usr/bin/env python3
"""Real-data walk-forward backtest of the FuelPricePredictor.

Pure-fade (numpy-only) model. This is the pre-shipping acceptance gate. It:

1. Fetches ~24 months of real FuelWatch ULP history (daily min price across WA).
2. Fits the real FuelPricePredictor on the full series and reads the in-fit
   train_metrics (numpy pure-fade walk-forward holdout vs the average baseline).
3. Runs an EXTENDED rolling-origin backtest over the last ~120 days: for each
   origin t (stepped weekly), fits on the prefix ending at t-3, forecasts 7
   days with known={t-1, t} as the live today/yesterday anchor, and scores
   MAE by horizon day (3..7), split into post-hike vs normal windows, for both
   the pure-fade model and the average baseline.
4. Prints cheapest-day hit rate (did the forecast's cheapest day match the
   actual cheapest day in the 7-day window?).
5. Prints an explicit ACCEPTANCE verdict against the gate:
      post_hike_mae <= 9.5 c/L  AND  overall MAE < baseline_mae.

Run from the repo root:
    .venv/bin/python tools/backtest.py
"""

from __future__ import annotations

import sys
import time
import urllib.request
from datetime import date
from pathlib import Path

import numpy as np

# Allow running as a script (no HA runtime needed) — mirrors download_history.py.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from custom_components.fuel_predictor_wa import historic_client  # noqa: E402
from custom_components.fuel_predictor_wa.predictor import (  # noqa: E402
    HIKE_ABS_FLOOR,
    FuelPricePredictor,
)

PRODUCT = "ULP"
MONTHS_TO_FETCH = 24
ROLLING_WINDOW_DAYS = 120
ROLLING_STEP_DAYS = 7
HORIZON = 7
# Horizon days that are pure forecast (1- and 2-based are the known anchors).
FORECAST_HORIZON_DAYS = list(range(3, HORIZON + 1))  # [3,4,5,6,7]
# Acceptance gate.
GATE_POST_HIKE_MAE = 9.5


# ---------------------------------------------------------------------------
# 1. Fetch + collapse to daily min series
# ---------------------------------------------------------------------------
def fetch_daily_series() -> tuple[list[date], list[float], int, int]:
    """Return (dates, min_prices, n_months_ok, n_months_attempted)."""
    today = date.today()
    months = historic_client.trailing_months(today, MONTHS_TO_FETCH)
    per_day: dict[date, list[float]] = {}
    ok = 0
    failed: list[str] = []
    t0 = time.time()
    for i, (y, m) in enumerate(months, 1):
        url = historic_client.month_url(y, m)
        label = f"{y}-{m:02d}"
        try:
            with urllib.request.urlopen(url, timeout=90) as resp:  # noqa: S310
                text = resp.read().decode("utf-8", errors="replace")
        except Exception as exc:  # noqa: BLE001
            failed.append(f"{label}({type(exc).__name__})")
            print(f"  [{i:>2}/{len(months)}] {label}: SKIP ({type(exc).__name__})")
            continue
        recs = historic_client.parse_csv(text, PRODUCT)
        for r in recs:
            per_day.setdefault(r["date"], []).append(r["price"])
        ok += 1
        print(
            f"  [{i:>2}/{len(months)}] {label}: {len(recs):>6} ULP rows  "
            f"(elapsed {time.time() - t0:.0f}s)"
        )
    dates = sorted(per_day.keys())
    series = [float(min(per_day[d])) for d in dates]
    if failed:
        print(f"  skipped months: {', '.join(failed)}")
    return dates, series, ok, len(months)


# ---------------------------------------------------------------------------
# 3. Rolling-origin backtest
# ---------------------------------------------------------------------------
def _baseline_predict(
    prefix_series: list[float],
    prefix_dates: list[date],
    target_date: date,
) -> float:
    """The OLD average-baseline math: recent_mean + (weekday_mean - overall_mean)."""
    recent = float(np.mean(prefix_series[-28:])) if prefix_series else 0.0
    overall = float(np.mean(prefix_series)) if prefix_series else 0.0
    by_wd: list[list[float]] = [[] for _ in range(7)]
    for d, p in zip(prefix_dates, prefix_series, strict=False):
        by_wd[d.weekday()].append(p)
    wd_means = [float(np.mean(xs)) if xs else overall for xs in by_wd]
    return recent + (wd_means[target_date.weekday()] - overall)


def rolling_origin_backtest(
    dates: list[date],
    series: list[float],
) -> dict:
    """Extended rolling-origin backtest over the last ROLLING_WINDOW_DAYS.

    Pure-fade production model (numpy-only, no global features).
    """
    n = len(series)
    if n < ROLLING_WINDOW_DAYS + HORIZON + 10:
        return {"error": f"series too short ({n} days)"}

    # Post-hike threshold using the FULL series (matches walk_forward semantics).
    diffs_all = np.diff(np.asarray(series, dtype=float))
    post_threshold = (
        max(HIKE_ABS_FLOOR, 1.0 * float(np.std(diffs_all))) if len(diffs_all) else HIKE_ABS_FLOOR
    )

    # Horizon-day accumulators: index = horizon day (3..7).
    new_mae_by_h: dict[int, list[float]] = {k: [] for k in FORECAST_HORIZON_DAYS}
    base_mae_by_h: dict[int, list[float]] = {k: [] for k in FORECAST_HORIZON_DAYS}
    # Post-hike vs normal (new model).
    new_post_hike: list[float] = []
    new_normal: list[float] = []
    base_post_hike: list[float] = []
    base_normal: list[float] = []
    # Cheapest-day hit tracking (over forecast days 3..7).
    cheapest_hits = 0
    cheapest_origins = 0
    # Overall (new + baseline) across all forecast days.
    new_all: list[float] = []
    base_all: list[float] = []

    origins_evaluated = 0
    last_origin = n - HORIZON  # need horizon days of actuals after the origin anchor
    first_origin = max(n - ROLLING_WINDOW_DAYS, 60)

    t0 = time.time()
    for t in range(first_origin, last_origin, ROLLING_STEP_DAYS):
        # Fit prefix: series[:t-2] (indices 0..t-3). Known anchors: t-1, t.
        if t - 2 < 40:
            continue
        prefix_dates = dates[: t - 2]
        prefix_series = series[: t - 2]
        prefix_dict = dict(zip(prefix_dates, prefix_series, strict=False))
        known = {dates[t - 1]: series[t - 1], dates[t]: series[t]}

        try:
            model = FuelPricePredictor()
            model.fit(prefix_dict)
            pts = model.predict(start=dates[t - 1], horizon=HORIZON, known=known)
        except Exception as exc:  # noqa: BLE001
            print(f"  origin t={t} ({dates[t]}): SKIP ({type(exc).__name__}: {exc})")
            continue

        # Map returned points by date for easy lookup.
        by_date = {p.day: p.price_cpl for p in pts}

        # Score each forecast horizon day (3..7 → dates[t+1] .. dates[t+5]).
        new_fc_prices: list[tuple[date, float, float]] = []  # (date, pred, actual)
        for k in FORECAST_HORIZON_DAYS:
            d = dates[t - 1 + (k - 1)]  # horizon day k (1-based)
            if d not in by_date or by_date[d] is None:
                continue
            di = dates.index(d)
            actual = float(series[di])
            pred_new = float(by_date[d])
            pred_base = _baseline_predict(prefix_series, prefix_dates, d)
            err_new = abs(actual - pred_new)
            err_base = abs(actual - pred_base)
            new_mae_by_h[k].append(err_new)
            base_mae_by_h[k].append(err_base)
            new_all.append(err_new)
            base_all.append(err_base)
            new_fc_prices.append((d, pred_new, actual))
            # Post-hike labelling on the actual day.
            is_post_hike = di >= 1 and (series[di] - series[di - 1]) > post_threshold
            if is_post_hike:
                new_post_hike.append(err_new)
                base_post_hike.append(err_base)
            else:
                new_normal.append(err_new)
                base_normal.append(err_base)

        # Cheapest-day hit (forecast days only — the actionable question).
        if len(new_fc_prices) >= 2:
            cheapest_origins += 1
            new_cheapest_date = min(new_fc_prices, key=lambda x: x[1])[0]
            actual_cheapest_date = min(new_fc_prices, key=lambda x: x[2])[0]
            if new_cheapest_date == actual_cheapest_date:
                cheapest_hits += 1

        origins_evaluated += 1

    elapsed = time.time() - t0
    return {
        "origins_evaluated": origins_evaluated,
        "elapsed_s": elapsed,
        "new_mae_by_h": {k: float(np.mean(v)) if v else None for k, v in new_mae_by_h.items()},
        "base_mae_by_h": {k: float(np.mean(v)) if v else None for k, v in base_mae_by_h.items()},
        "new_overall_mae": float(np.mean(new_all)) if new_all else None,
        "base_overall_mae": float(np.mean(base_all)) if base_all else None,
        "new_post_hike_mae": float(np.mean(new_post_hike)) if new_post_hike else None,
        "new_normal_mae": float(np.mean(new_normal)) if new_normal else None,
        "base_post_hike_mae": float(np.mean(base_post_hike)) if base_post_hike else None,
        "base_normal_mae": float(np.mean(base_normal)) if base_normal else None,
        "n_post_hike": len(new_post_hike),
        "n_normal": len(new_normal),
        "cheapest_hits": cheapest_hits,
        "cheapest_origins": cheapest_origins,
        "post_threshold": post_threshold,
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def _fmt(x: float | None, places: int = 2) -> str:
    return f"{x:.{places}f}" if x is not None else "n/a"


def main() -> int:
    print("=" * 72)
    print("FuelPricePredictor — real-data backtest (pure-fade, numpy-only)")
    print("=" * 72)

    print(f"\n[1] Fetching {MONTHS_TO_FETCH} months of FuelWatch ULP history...")
    dates, series, months_ok, months_attempted = fetch_daily_series()
    if months_ok < 6:
        print(
            f"\nFAIL: only {months_ok} months fetched — need >= 6 for the fade tier. "
            "Re-run when the network/FuelWatch is available."
        )
        return 2
    span_days = (dates[-1] - dates[0]).days if len(dates) > 1 else 0
    print(
        f"\n  got {months_ok}/{months_attempted} months -> {len(dates)} daily points, "
        f"span {dates[0]} .. {dates[-1]} ({span_days} days), "
        f"price range {min(series):.1f}..{max(series):.1f} c/L"
    )

    print("\n[2] Fitting FuelPricePredictor on the full series...")
    t0 = time.time()
    model = FuelPricePredictor()
    model.fit(dict(zip(dates, series, strict=False)))
    fit_s = time.time() - t0
    tm = model.train_metrics
    print(f"  fit in {fit_s:.2f}s  (model_kind={tm.get('model_kind')})")
    print(
        f"  cycle_len_days={tm.get('cycle_len_days')}  "
        f"n_hikes={tm.get('n_hikes')}  n_train={tm.get('n_train')}  "
        f"n_holdout={tm.get('n_holdout')}"
    )

    print("\n  in-fit walk-forward train_metrics (full-series fit):")
    print(f"    mae             = {_fmt(tm.get('mae'))} c/L")
    print(f"    baseline_mae    = {_fmt(tm.get('baseline_mae'))} c/L  (weekday_mean + recent_mean)")
    print(f"    improvement_pct = {_fmt(tm.get('improvement_pct'))} %")
    print(f"    mape_pct        = {_fmt(tm.get('mape_pct'))} %")
    print(f"    post_hike_mae   = {_fmt(tm.get('post_hike_mae'))} c/L")
    print(f"    normal_mae      = {_fmt(tm.get('normal_mae'))} c/L")
    print(f"    beats_baseline  = {tm.get('beats_baseline')}")

    print(
        f"\n[3] Rolling-origin backtest — pure-fade model "
        f"(last {ROLLING_WINDOW_DAYS} days, step {ROLLING_STEP_DAYS}d, horizon {HORIZON}d)..."
    )
    rb = rolling_origin_backtest(dates, series)
    if "error" in rb:
        print(f"  FAIL: {rb['error']}")
        return 3
    print(
        f"  origins evaluated = {rb['origins_evaluated']}  "
        f"(post-hike threshold = {rb['post_threshold']:.2f} c/L, "
        f"{rb['n_post_hike']} post-hike / {rb['n_normal']} normal forecast-days, "
        f"{rb['elapsed_s']:.1f}s)"
    )

    print("\n  MAE by horizon day (c/L) — pure-fade vs average baseline:")
    print(f"    {'day':>4} {'date_off':>8} {'model':>8} {'base':>8} {'delta':>8}")
    for k in FORECAST_HORIZON_DAYS:
        nm = rb["new_mae_by_h"].get(k)
        bm = rb["base_mae_by_h"].get(k)
        delta = (nm - bm) if (nm is not None and bm is not None) else None
        print(f"    {k:>4} {f'+{k - 2}d':>8} {_fmt(nm):>8} {_fmt(bm):>8} {_fmt(delta):>8}")

    hit_rate = (
        rb["cheapest_hits"] / rb["cheapest_origins"] * 100.0 if rb["cheapest_origins"] else None
    )
    print(
        f"\n  overall_mae={_fmt(rb['new_overall_mae'])} c/L  "
        f"post_hike_mae={_fmt(rb['new_post_hike_mae'])} c/L  "
        f"cheapest_day_hit_rate={_fmt(hit_rate, 1)} %  "
        f"({rb['cheapest_hits']}/{rb['cheapest_origins']})"
    )
    print("  (random reference for 5 forecast days = 20.0 %)")
    print("=" * 72)

    # ---- Acceptance verdict ------------------------------------------------
    gate_post = tm.get("post_hike_mae")
    gate_beats = tm.get("beats_baseline")
    post_ok = gate_post is not None and gate_post <= GATE_POST_HIKE_MAE
    beats_ok = bool(gate_beats)
    verdict = "PASS" if (post_ok and beats_ok) else "FAIL"

    print("\n" + "=" * 72)
    print("ACCEPTANCE VERDICT (pure-fade vs average baseline)")
    print("=" * 72)
    print(f"  gate: post_hike_mae <= {GATE_POST_HIKE_MAE} c/L  AND  beats_baseline == True")
    print(f"  in-fit post_hike_mae   = {_fmt(gate_post)} c/L   -> {'OK' if post_ok else 'MISS'}")
    print(f"  in-fit beats_baseline  = {gate_beats}            -> {'OK' if beats_ok else 'MISS'}")
    print(f"  rolling overall model  = {_fmt(rb['new_overall_mae'])} c/L")
    print(f"  rolling overall base   = {_fmt(rb['base_overall_mae'])} c/L")
    print(f"\n  >>> VERDICT: {verdict} <<<")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
