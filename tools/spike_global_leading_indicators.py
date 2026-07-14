"""OFFLINE spike: do global leading indicators improve the WA fuel forecast?

Not shipped to HA. Run in the dev venv:
    .venv/bin/python tools/spike_global_leading_indicators.py

Tests whether adding a level-drift term — ``β · recent_global_return`` — to the
pure-fade forecast beats plain fade on a walk-forward holdout. Fits β (pass-
through elasticity) on a train split, applies on holdout, per indicator
(RBOB / Brent / AUD-USD) and lag window (7 / 10 / 14 d). Faithful to the
production ``_walk_forward`` (anchor at h-2, fade from prefix, >=3 hikes, clamp).
"""

from __future__ import annotations

import csv
import json
import sys
import urllib.request
from datetime import date, timedelta
from pathlib import Path
from statistics import mean

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from custom_components.fuel_predictor_wa.const import YAHOO_BASE_URL
from custom_components.fuel_predictor_wa.global_features import parse_chart  # pure, no HA
from custom_components.fuel_predictor_wa.historic_client import month_url, parse_csv
from custom_components.fuel_predictor_wa.predictor import (  # noqa: E402
    CLAMP_HI,
    CLAMP_LO,
    MIN_LEVEL_WINDOW,
    _fade_curve_for,
    cycle_pos_at,
    detect_hikes,
    median_cycle_len,
)

PRODUCT = "PULP"  # Premium 95
MONTHS = 24
UA = "ha-fuel-predictor-wa/0.1 (+https://github.com/madeinoz67/ha-fuel-predictor-wa)"
BLOB_CACHE = Path("/tmp/fw_cache")
GLOBAL_CACHE = Path("/tmp/yh_cache")


def fetch_wa_series() -> list[tuple[date, float]]:
    """24 months of WA-wide PULP min/day (same source as production training)."""
    today = date.today()
    y, m = today.year, today.month
    by_date: dict[date, float] = {}
    for _ in range(MONTHS):
        cached = BLOB_CACHE / f"{y}-{m:02d}.csv"
        if cached.exists():
            text = cached.read_text()
        else:
            BLOB_CACHE.mkdir(parents=True, exist_ok=True)
            with urllib.request.urlopen(month_url(y, m), timeout=60) as resp:  # noqa: S310
                text = resp.read().decode()
            cached.write_text(text)
        for r in parse_csv(text, PRODUCT):
            d, p = r["date"], r["price"]
            if d not in by_date or p < by_date[d]:
                by_date[d] = p
        m -= 1
        if m == 0:
            m, y = 12, y - 1
    return sorted(by_date.items())


WHOLESALE_BLOB = "https://warsydprdstafuelwatch.blob.core.windows.net/historical-reports"


def fetch_tgp_series() -> dict[date, float]:
    """WA PULP Terminal Gate Price min/day from the FuelWatch wholesale CSVs.

    Same blob/Mogas-derived wholesale price WA retailers pay; free + same source
    as the retail data. Yearly files (FuelWatchWholesale-{yyyy}.csv). NB: the
    wholesale CSV uses PRODUCT (not PRODUCT_DESCRIPTION), so it has its own parse.
    """
    today = date.today()
    years = [today.year, today.year - 1, today.year - 2]
    by_date: dict[date, float] = {}
    for yr in years:
        cached = BLOB_CACHE / f"wholesale-{yr}.csv"
        if cached.exists():
            text = cached.read_text()
        else:
            BLOB_CACHE.mkdir(parents=True, exist_ok=True)
            with urllib.request.urlopen(f"{WHOLESALE_BLOB}/FuelWatchWholesale-{yr}.csv", timeout=60) as resp:  # noqa: S310
                text = resp.read().decode()
            cached.write_text(text)
        for row in csv.DictReader(text.splitlines()):
            if row.get("PRODUCT") != PRODUCT:
                continue
            try:
                dd, mm, yyyy = (int(x) for x in row["PUBLISH_DATE"].split("/"))
                d, p = date(yyyy, mm, dd), float(row["PRODUCT_PRICE"])
            except (KeyError, ValueError, TypeError):
                continue
            if d not in by_date or p < by_date[d]:
                by_date[d] = p
    return dict(sorted(by_date.items()))


def fetch_global(symbol: str) -> dict[date, float]:
    """Yahoo daily closes for one symbol (cached)."""
    GLOBAL_CACHE.mkdir(parents=True, exist_ok=True)
    cached = GLOBAL_CACHE / f"{symbol.replace('=', '_')}.json"
    if cached.exists():
        return {date.fromisoformat(k): v for k, v in __import__("json").loads(cached.read_text()).items()}
    url = f"{YAHOO_BASE_URL}/{symbol}?range=2y&interval=1d"
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
        text = resp.read().decode()
    parsed = parse_chart(text)
    cached.write_text(json.dumps({d.isoformat(): v for d, v in parsed.items()}))
    return parsed


def forward_fill(global_by_date: dict[date, float], wa_dates: list[date]) -> dict[date, float]:
    """Carry global close forward onto each WA date (most-recent <= that date)."""
    filled: dict[date, float] = {}
    gitems = sorted(global_by_date.items())
    last = None
    # walk WA dates in order, advancing the global pointer
    gptr = 0
    for d in wa_dates:
        while gptr < len(gitems) and gitems[gptr][0] <= d:
            last = gitems[gptr][1]
            gptr += 1
        if last is not None:
            filled[d] = last
    return filled


def global_returns(filled: dict[date, float], dates: list[date], lag: int) -> dict[date, float]:
    """{date: pct change over the previous `lag` days} (causal)."""
    out: dict[date, float] = {}
    for i, d in enumerate(dates):
        if i < lag:
            continue
        past = dates[i - lag]
        cur = filled.get(d)
        old = filled.get(past)
        if cur and old and old != 0:
            out[d] = cur / old - 1.0
    return out


def fade_pred(prices: list[float], h: int, clamp_lo: float, clamp_hi: float) -> float | None:
    """Faithful single-point pure-fade prediction of prices[h]."""
    anchor_t = h - 2
    if anchor_t < MIN_LEVEL_WINDOW + 2:
        return None
    prefix = prices[:anchor_t]
    hikes = detect_hikes(prefix)
    if len(hikes) < 3:
        return None
    L = median_cycle_len(hikes)
    fade, fade_mean = _fade_curve_for(prefix, hikes, L)
    if not fade:
        return None
    anchor_price = prices[anchor_t]
    a_cp = cycle_pos_at(anchor_t, hikes) % L
    t_cp = cycle_pos_at(h, hikes) % L
    pred = anchor_price + (fade.get(t_cp, fade_mean) - fade.get(a_cp, fade_mean))
    return max(clamp_lo, min(clamp_hi, pred))


def run() -> None:
    print("Loading WA PULP (24mo)...", flush=True)
    wa = fetch_wa_series()
    dates = [d for d, _ in wa]
    prices = [p for _, p in wa]
    n = len(prices)
    clamp_lo = CLAMP_LO * min(prices[-28:])
    clamp_hi = CLAMP_HI * max(prices[-28:])
    print(f"  {n} days. Fetching global indicators...", flush=True)

    indicators = {
        "TGP (WA wholesale)": fetch_tgp_series(),
        "RBOB (RB=F)": fetch_global("RB=F"),
        "Brent (BZ=F)": fetch_global("BZ=F"),
        "AUD/USD (AUDUSD=X)": fetch_global("AUDUSD=X"),
    }

    # Precompute plain-fade pred + residual for a sample of days (train + holdout).
    start = MIN_LEVEL_WINDOW + 4
    step = 4
    sample_idx = list(range(start, n, step))
    plain_pred: dict[int, float] = {}
    for h in sample_idx:
        p = fade_pred(prices, h, clamp_lo, clamp_hi)
        if p is not None:
            plain_pred[h] = p
    scored = sorted(plain_pred)
    if len(scored) < 40:
        print(f"  only {len(scored)} fade-scorable days — too few")
        return
    split = int(len(scored) * 0.7)
    train_idx = scored[:split]
    hold_idx = scored[split:]
    print(f"  train {len(train_idx)} / holdout {len(hold_idx)} days\n", flush=True)

    # Plain-fade holdout MAE (baseline).
    plain_mae = mean(abs(prices[h] - plain_pred[h]) for h in hold_idx)
    print(f"=== Plain-fade holdout MAE: {plain_mae:.2f} c/L  (baseline)\n")

    print(f"{'indicator':<22}{'lag':>5}{'β':>8}{'drift MAE':>11}{'Δ vs plain':>13}{'verdict':>10}")
    for name, gseries in indicators.items():
        if not gseries:
            print(f"{name:<22}  (no data)")
            continue
        filled = forward_fill(gseries, dates)
        for lag in (7, 10, 14):
            gret = global_returns(filled, dates, lag)
            # Fit β on TRAIN: least-squares slope through origin of residual on gret.
            num = sum((prices[h] - plain_pred[h]) * gret.get(dates[h - 2], 0.0) for h in train_idx)
            den = sum(gret.get(dates[h - 2], 0.0) ** 2 for h in train_idx)
            beta = num / den if den else 0.0
            # Apply on HOLDOUT.
            drift_errs = [
                abs(prices[h] - (plain_pred[h] + beta * gret.get(dates[h - 2], 0.0))) for h in hold_idx
            ]
            drift_mae = mean(drift_errs)
            delta = drift_mae - plain_mae
            verdict = "HELPS" if delta < -0.05 else ("hurts" if delta > 0.05 else "wash")
            print(f"{name:<22}{lag:>5}{beta:>8.2f}{drift_mae:>11.2f}{delta:>+13.2f}{verdict:>10}")
        print()


if __name__ == "__main__":
    run()
