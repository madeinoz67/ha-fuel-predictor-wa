"""OFFLINE accuracy spike v2: local-vs-WA, with a methodology self-check + the
cheapest-day metric (the production use case). Not shipped to HA.

    .venv/bin/python tools/accuracy_spike_local_vs_wa.py

SELF-CHECK: scoring WA-fade against WA actuals must reproduce the production
holdout MAE (~2.7 c/L). If it doesn't, the harness is wrong and nothing below it
is trustworthy.

METRICS:
  - price MAE: |pred - actual| per holdout day, for (WA-on-WA), (WA->local),
    (local->local). Controlled: WA->local and local->local both anchor to the
    SAME local price, so only the fade-curve source differs.
  - cheapest-day hit rate: for each holdout 'today', forecast 7 days anchored to
    local[today]; does the predicted cheapest day match the actual cheapest day?
  - baseline: weekday-mean + recent-mean (the production baseline, ~9 c/L), NOT
    day-ago (which is trivially good at price level and irrelevant here).

CATCHMENT: AREA='Bunbury' is HARDCODED HERE ONLY to run the experiment on
Stephen's data. The v0.3.0 integration resolves this dynamically from the
configured suburb + radius.
"""

from __future__ import annotations

import csv
import sys
import urllib.request
from datetime import date, timedelta
from pathlib import Path
from statistics import mean

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from custom_components.fuel_predictor_wa.predictor import (  # noqa: E402
    CLAMP_HI,
    CLAMP_LO,
    MIN_LEVEL_WINDOW,
    _fade_curve_for,
    cycle_pos_at,
    detect_hikes,
    median_cycle_len,
)

PRODUCT = "PULP"
AREA = "Bunbury"
MONTHS = 24
HORIZON = 7
CACHE = Path("/tmp/fw_cache")
CSV = Path("/tmp/fw_jul.csv")


def _parse_date(value: str) -> date:
    dd, mm, yyyy = (int(x) for x in value.split("/"))
    return date(yyyy, mm, dd)


def fetch_month(year: int, month: int) -> str:
    CACHE.mkdir(parents=True, exist_ok=True)
    cached = CACHE / f"{year}-{month:02d}.csv"
    if cached.exists():
        return cached.read_text()
    url = (
        "https://warsydprdstafuelwatch.blob.core.windows.net/historical-reports/"
        f"FuelWatchRetail-{month:02d}-{year}.csv"
    )
    with urllib.request.urlopen(url, timeout=60) as resp:  # noqa: S310
        text = resp.read().decode()
    cached.write_text(text)
    return text


def load_records() -> list[dict]:
    records: list[dict] = []
    today = date.today()
    y, m = today.year, today.month
    for _ in range(MONTHS):
        if y == today.year and m == today.month and CSV.exists():
            text = CSV.read_text()
        else:
            text = fetch_month(y, m)
        for row in csv.DictReader(text.splitlines()):
            if row.get("PRODUCT_DESCRIPTION") != PRODUCT:
                continue
            try:
                records.append(
                    {
                        "date": _parse_date(row["PUBLISH_DATE"]),
                        "price": float(row["PRODUCT_PRICE"]),
                        "suburb": row.get("LOCATION"),
                        "address": row.get("ADDRESS"),
                        "area": row.get("AREA_DESCRIPTION"),
                        "region": row.get("REGION_DESCRIPTION"),
                    }
                )
            except (KeyError, ValueError, TypeError):
                continue
        m -= 1
        if m == 0:
            m = 12
            y -= 1
    return records


def min_per_day(
    records: list[dict], area_filter: str | None = None, region_filter: str | None = None
) -> dict[date, float]:
    by_date: dict[date, float] = {}
    for r in records:
        if area_filter is not None and r["area"] != area_filter:
            continue
        if region_filter is not None and r["region"] != region_filter:
            continue
        d, p = r["date"], r["price"]
        if d not in by_date or p < by_date[d]:
            by_date[d] = p
    return dict(sorted(by_date.items()))


def count_stations(records: list[dict], area_filter, region_filter) -> int:
    """Distinct (suburb,address) trading in the most recent 30 days — station count."""
    latest = max(r["date"] for r in records)
    cutoff = latest - timedelta(days=30)
    seen = {
        (r["suburb"], r["address"])
        for r in records
        if r["date"] >= cutoff
        and (area_filter is None or r["area"] == area_filter)
        and (region_filter is None or r["region"] == region_filter)
    }
    return len(seen)


def _clamp(series_full):
    return CLAMP_LO * min(series_full[-28:]), CLAMP_HI * max(series_full[-28:])


def walk_forward(source, target, hold, c_lo, c_hi, step=None):
    """Faithful mirror of predictor._walk_forward.

    fade curve from `source`; anchor + scoring against `target`. source==target
    reproduces production. Returns (mae_or_None, n_scored, errors).
    Mirrors production: prefix=src[:h-2], anchor=src[h-2], predict h, skip when
    prefix has <3 hikes, clamp from the FULL series tail. `step` defaults to
    production's hold//7 (~7 refits); pass a smaller step for a powered study.
    """
    n = len(source)
    if hold < 3 or n < MIN_LEVEL_WINDOW + hold + 3:
        return None, 0, []
    step = max(1, hold // 7) if step is None else step
    errs: list[float] = []
    for h in range(n - hold, n, step):
        anchor_t = h - 2
        if anchor_t < MIN_LEVEL_WINDOW + 2:
            continue
        prefix = source[:anchor_t]
        hikes = detect_hikes(prefix)
        if len(hikes) < 3 or not hikes:
            continue
        L = median_cycle_len(hikes)
        fade, fade_mean = _fade_curve_for(prefix, hikes, L)
        if not fade:
            continue
        # Controlled: anchor to the TARGET price (what we're predicting), so the
        # only thing that differs between models is the fade curve's source.
        # When source==target (self-check) this is identical to production.
        anchor_price = target[anchor_t]
        a_cp = cycle_pos_at(anchor_t, hikes) % L
        t_cp = cycle_pos_at(h, hikes) % L
        pred = anchor_price + (fade.get(t_cp, fade_mean) - fade.get(a_cp, fade_mean))
        pred = max(c_lo, min(c_hi, pred))
        errs.append(abs(target[h] - pred))
    return (float(np.mean(errs)) if errs else None), len(errs), errs


def horizon_hits(source, target, hold, c_lo, c_hi, days=HORIZON, step=None):
    """Cheapest-day hit rate: fade from `source`, anchor+actual from `target`."""
    n = len(source)
    step = max(1, hold // 7) if step is None else step
    exact = within1 = scored = 0
    for t in range(n - hold, n - days, step):
        anchor_price = target[t]
        prefix = source[:t]
        hikes = detect_hikes(prefix)
        if len(hikes) < 3 or not hikes:
            continue
        L = median_cycle_len(hikes)
        fade, fade_mean = _fade_curve_for(prefix, hikes, L)
        if not fade:
            continue
        a_cp = cycle_pos_at(t, hikes) % L
        a_fade = fade.get(a_cp, fade_mean)
        preds = {t: anchor_price}
        for i in range(1, days + 1):
            t_cp = cycle_pos_at(t + i, hikes) % L
            preds[t + i] = max(c_lo, min(c_hi, anchor_price + (fade.get(t_cp, fade_mean) - a_fade)))
        actual = {t + i: target[t + i] for i in range(days + 1)}
        cp = min(preds, key=preds.get)
        ca = min(actual, key=actual.get)
        exact += cp == ca
        within1 += abs(cp - ca) <= 1
        scored += 1
    return exact, within1, scored


def weekday_mean_baseline(series: list[float], dates: list[date]) -> list[float]:
    """Production baseline: weekday_mean + recent_mean shift, per index."""
    n = len(series)
    out = [float("nan")] * n
    for h in range(MIN_LEVEL_WINDOW + 2, n):
        prefix = series[: h - 2]
        by_wd = [[] for _ in range(7)]
        for d, p in zip(dates[: h - 2], prefix, strict=False):
            by_wd[d.weekday()].append(p)
        wd_means = [mean(x) if x else mean(prefix) for x in by_wd]
        recent = mean(prefix[-28:])
        overall = mean(prefix)
        out[h] = recent + (wd_means[dates[h].weekday()] - overall)
    return out


def run() -> None:
    print(f"Loading {MONTHS} months of {PRODUCT}...", flush=True)
    records = load_records()
    wa = list(min_per_day(records, None).items())
    loc = list(min_per_day(records, AREA).items())
    wa_d = dict(wa)
    loc_d = dict(loc)
    common = sorted(set(wa_d) & set(loc_d))
    dates = common
    wa_p = [wa_d[d] for d in common]
    lo_p = [loc_d[d] for d in common]
    print(f"  {len(records):,} station-days | WA {len(wa)}d | local {len(loc)}d | aligned {len(common)}d\n")

    bunbury = list(min_per_day(records, area_filter=AREA).items())
    bun_d = dict(bunbury)
    bun_dates = list(bun_d)
    bun_p = [bun_d[d] for d in bun_dates]
    bn = len(bun_p)

    # Self-check at production params — must be ~2.7.
    wa_all = list(min_per_day(records).items())
    wa_d = dict(wa_all)
    wcommon = sorted(set(wa_d) & set(bun_d))
    wa_p = [wa_d[d] for d in wcommon]
    bp = [bun_d[d] for d in wcommon]
    hold_sc = min(28, len(wcommon) // 5)
    wa_mae_sc, _, _ = walk_forward(wa_p, wa_p, hold_sc, *_clamp(wa_p))
    print("=== SELF-CHECK (methodology validity) ===")
    print(f"  WA fade scored on WA actuals : {wa_mae_sc:.2f} c/L  (production ~2.7)\n")

    hold = min(180, len(wcommon) - MIN_LEVEL_WINDOW - 10)
    step = max(1, hold // 30)

    # Sweep catchments from local outward. Each is scored PREDICTING BUNBURY,
    # anchored to Bunbury — so the only variable is the fade curve's source data.
    catchments = [
        ("Bunbury area", AREA, None),
        ("South-West region", None, "South-West"),
        ("WA-wide (Perth-skewed)", None, None),
    ]
    print(f"=== CATCHMENT SWEEP (predict Bunbury; anchor=Bunbury; hold={hold}d, ~30 pts) ===")
    print(f"  {'catchment':<26}{'stations':>9}{'price MAE':>11}{'cheap exact':>13}{'within±1d':>11}")
    for label, af, rf in catchments:
        c_d = min_per_day(records, area_filter=af, region_filter=rf)
        common = sorted(set(c_d) & set(bun_d))
        if len(common) < MIN_LEVEL_WINDOW + hold + 3:
            print(f"  {label:<26}{'-':>9} (too few common days)")
            continue
        cs = [c_d[d] for d in common]
        ts = [bun_d[d] for d in common]
        clo, chi = _clamp(cs)
        mae, mae_n, _ = walk_forward(cs, ts, hold, clo, chi, step)
        ex, w1, sc = horizon_hits(cs, ts, hold, clo, chi, step=step)
        n_st = count_stations(records, af, rf)
        mae_s = f"{mae:.2f}" if mae is not None else "-"
        ex_s = f"{100*ex/sc:.0f}%" if sc else "-"
        w1_s = f"{100*w1/sc:.0f}%" if sc else "-"
        print(f"  {label:<26}{n_st:>9}{mae_s:>11}{ex_s:>13}{w1_s:>11}")


if __name__ == "__main__":
    run()
