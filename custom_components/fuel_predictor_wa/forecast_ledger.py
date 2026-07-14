"""Forecast accuracy ledger: snapshot daily forecasts + actuals, score accuracy.

The live coordinator overwrites its forecast every poll and stores no history,
so a true forecast-vs-actual accuracy curve is impossible from coordinator state
alone (see project notes / data-gap audit). This module is the instrumentation
that closes that gap:

  - ``append_forecast`` — idempotent per issued day; records each horizon point
    the model predicted (with its target date + days-out).
  - ``append_actual``   — idempotent upsert per date; records the realised
    cheapest c/L once a day arrives.
  - ``load_accuracy``   — joins the two on ``target_date == actual_date`` and
    returns MAE overall + MAE-by-days-out (how accuracy degrades with horizon),
    plus recent pairs for a scatter/strip view.

CSVs live in the per-entry storage dir next to ``model.pkl``. All functions are
pure file I/O (safe to run in the executor) and best-effort: the coordinator
wraps them so a ledger failure never breaks forecasting.
"""

from __future__ import annotations

import csv
import logging
from datetime import date, timedelta
from pathlib import Path
from statistics import mean
from typing import Any

_LOGGER = logging.getLogger(__name__)

FORECAST_LEDGER_FILENAME = "forecast_ledger.csv"
ACTUALS_LEDGER_FILENAME = "actuals.csv"
MAX_LEDGER_AGE_DAYS = 180

_FORECAST_HEADER = ["issued_date", "target_date", "days_out", "predicted_cpl", "source"]
_ACTUAL_HEADER = ["date", "actual_cpl"]


def _issued_dates(path: Path) -> set[str]:
    """Return the set of issued_date strings already in the forecast ledger."""
    if not path.exists():
        return set()
    dates: set[str] = set()
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            d = row.get("issued_date")
            if d:
                dates.add(d)
    return dates


def append_forecast(storage_dir: Path, issued_date: date, points: list) -> None:
    """Append this issued day's horizon to the ledger (idempotent per issued day).

    ``points`` is a list of ``DayForecast`` (``.day``, ``.price_cpl``, ``.source``).
    Rows with a null price (unfitted model) are skipped — there is nothing to score.
    """
    storage_dir = Path(storage_dir)
    storage_dir.mkdir(parents=True, exist_ok=True)
    path = storage_dir / FORECAST_LEDGER_FILENAME
    if issued_date.isoformat() in _issued_dates(path):
        return
    new_file = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        if new_file:
            writer.writerow(_FORECAST_HEADER)
        for p in points:
            price = getattr(p, "price_cpl", None)
            if price is None:
                continue
            target = getattr(p, "day", None)
            if target is None:
                continue
            days_out = max(0, (target - issued_date).days)
            writer.writerow(
                [
                    issued_date.isoformat(),
                    target.isoformat(),
                    days_out,
                    f"{float(price):.1f}",
                    getattr(p, "source", "forecast"),
                ]
            )


def append_actual(storage_dir: Path, actual_date: date, actual_cpl: float | None) -> None:
    """Record (or overwrite) the realised cheapest c/L for ``actual_date``.

    Idempotent upsert keyed on date: re-polled days update the value rather than
    duplicate. A null ``actual_cpl`` is a no-op (nothing observed yet).
    """
    if actual_cpl is None:
        return
    storage_dir = Path(storage_dir)
    storage_dir.mkdir(parents=True, exist_ok=True)
    path = storage_dir / ACTUALS_LEDGER_FILENAME
    rows: dict[str, str] = {}
    if path.exists():
        with path.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                d = row.get("date")
                if d:
                    rows[d] = row.get("actual_cpl", "")
    rows[actual_date.isoformat()] = f"{float(actual_cpl):.1f}"
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(_ACTUAL_HEADER)
        for d in sorted(rows):
            writer.writerow([d, rows[d]])


def load_accuracy(storage_dir: Path, max_age_days: int = MAX_LEDGER_AGE_DAYS) -> dict:
    """Join forecasts with actuals and return accuracy stats for the dashboard.

    Returns::
        {
          overall_mae: float | None,          # MAE across all paired (pred, actual)
          n_pairs: int,                       # count of scored forecast/actual pairs
          mae_by_days_out: [{days_out, mae, n}],  # accuracy degradation with horizon
          recent: [{target_date, issued_date, days_out, predicted, actual, error}],  # last ~30
          coverage_forecast_days: int,        # distinct days a forecast was issued
          coverage_actual_days: int,          # distinct days an actual was recorded
          bias: float | None,                 # mean(predicted - actual), signed
        }

    Empty/stub result when no forecast ledger exists yet (fresh install).
    """
    storage_dir = Path(storage_dir)
    fpath = storage_dir / FORECAST_LEDGER_FILENAME
    apath = storage_dir / ACTUALS_LEDGER_FILENAME
    stub: dict[str, Any] = {
        "overall_mae": None,
        "n_pairs": 0,
        "mae_by_days_out": [],
        "recent": [],
        "coverage_forecast_days": 0,
        "coverage_actual_days": 0,
        "bias": None,
    }
    if not fpath.exists():
        return stub

    actuals: dict[str, float] = {}
    if apath.exists():
        with apath.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                try:
                    actuals[row["date"]] = float(row["actual_cpl"])
                except (ValueError, TypeError, KeyError):
                    continue

    today = date.today()
    cutoff = today - timedelta(days=max_age_days)

    pairs: list[dict] = []
    by_days: dict[int, list[float]] = {}
    signed_errors: list[float] = []
    forecast_days: set[str] = set()

    with fpath.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            issued = row.get("issued_date")
            if issued:
                forecast_days.add(issued)
            target_s = row.get("target_date")
            if not target_s:
                continue
            try:
                target = date.fromisoformat(target_s)
                predicted = float(row["predicted_cpl"])
                days_out = int(row["days_out"])
            except (ValueError, TypeError, KeyError):
                continue
            if target < cutoff:
                continue
            if target_s not in actuals:
                continue
            actual = actuals[target_s]
            error = abs(predicted - actual)
            pairs.append(
                {
                    "target_date": target_s,
                    "issued_date": issued,
                    "days_out": days_out,
                    "predicted": round(predicted, 1),
                    "actual": round(actual, 1),
                    "error": round(error, 2),
                }
            )
            by_days.setdefault(days_out, []).append(error)
            signed_errors.append(predicted - actual)

    if not pairs:
        stub["coverage_forecast_days"] = len(forecast_days)
        stub["coverage_actual_days"] = len(actuals)
        return stub

    errors = [p["error"] for p in pairs]
    mae_by_days_out = sorted(
        [{"days_out": d, "mae": round(mean(es), 2), "n": len(es)} for d, es in by_days.items()],
        key=lambda x: x["days_out"],
    )
    recent = list(sorted(pairs, key=lambda p: (p["target_date"], p["days_out"]))[-30:])
    return {
        "overall_mae": round(mean(errors), 2),
        "n_pairs": len(pairs),
        "mae_by_days_out": mae_by_days_out,
        "recent": recent,
        "coverage_forecast_days": len(forecast_days),
        "coverage_actual_days": len(actuals),
        "bias": round(mean(signed_errors), 2),
    }
