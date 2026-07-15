"""Cycle-aware pure-fade fuel-price forecaster (numpy-only).

The production forecast is an empirical-fade curve anchored to the live known
price:

    forecast(day) = anchor_price + (fade[day_cp] - fade[anchor_cp])

where ``fade[cp]`` is the mean historical price at cycle-position ``cp``. This
kills the post-hike bias that an average-baseline suffers (it under-forecasts
post-hike by ~the recent_mean-to-peak gap, ~19c/L on WA's weekly cycle) because
the LEVEL is pinned by the live anchor while the SHAPE comes from the observed
cycle.

No third-party ML libraries are used. An earlier gradient-boosting / ridge
regression path was vestigial: ML-6 found the pure-fade forecast beats it, and
that regressor was only fit for the comparison metric, never used in
``predict``. Dropping it leaves the forecast unchanged; only the train_metrics
computation (now a numpy pure-fade holdout) differs. This also resolves the
install failure on Home Assistant's Python 3.14, where no prebuilt wheel for
the ML dependency is available and its source build fails.

Public contract preserved:

  - class FuelPricePredictor with fit / predict / cheapest
  - DayForecast(day, price_cpl, source)
  - ForecastResult(points, cheapest_day) with .cheapest_price
  - the _fitted flag
"""

from __future__ import annotations

import bisect
import logging
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from statistics import mean

import numpy as np

_LOGGER = logging.getLogger(__name__)

# --- Cycle / feature constants ----------------------------------------------
DEFAULT_CYCLE_LEN = 7
MIN_CYCLE_LEN = 4
MAX_CYCLE_LEN = 14

HIKE_ABS_FLOOR = 1.5  # cents; diffs below this are never hikes
HIKE_ROLLING_WINDOW = 14

TRAILING_LEN = 28  # level feature window + clamp window
MIN_LEVEL_WINDOW = 7  # min rows before features are defined
VOLATILITY_WINDOW = 7

CLAMP_LO = 0.85  # clamp factor on min28
CLAMP_HI = 1.15  # clamp factor on max28

TGP_LAG_DAYS = 10  # wholesale TGP → retail pass-through window (drift term)


# --- Public dataclasses (contract) ------------------------------------------
@dataclass
class DayForecast:
    """One day of the horizon."""

    day: date
    price_cpl: float | None  # cents per litre; None when no history yet
    source: str  # "known" | "forecast"


@dataclass
class ForecastResult:
    """A complete horizon + the cheapest day within it."""

    points: list[DayForecast]
    cheapest_day: DayForecast

    @property
    def cheapest_price(self) -> float | None:
        return self.cheapest_day.price_cpl


# --- Pure-numpy cycle primitives --------------------------------------------
def _rolling_std(xs: np.ndarray, window: int) -> np.ndarray:
    """Rolling population std over a shrinking window at the start.

    Returns same length as ``xs``; element i is std(xs[max(0,i-window+1):i+1]).
    """
    xs = np.asarray(xs, dtype=float)
    n = len(xs)
    out = np.zeros(n, dtype=float)
    for i in range(n):
        lo = max(0, i - window + 1)
        out[i] = float(np.std(xs[lo : i + 1]))
    return out


def detect_hikes(prices: list[float]) -> list[int]:
    """Return indices of hike-days in ``prices``.

    A hike-day at index ``i`` (i >= 1) means ``prices[i] - prices[i-1]`` is a
    prominent positive spike:

      - diff > 0
      - diff > max(HIKE_ABS_FLOOR, rolling_std_14(diff at i))
      - diff is the largest positive diff in a +-1 window (local prominence)
    """
    n = len(prices)
    if n < 2:
        return []
    arr = np.asarray(prices, dtype=float)
    diffs = np.diff(arr)  # length n-1; diffs[k] = arr[k+1] - arr[k]
    roll_std = _rolling_std(diffs, HIKE_ROLLING_WINDOW)
    hikes: list[int] = []
    for k in range(len(diffs)):
        d = float(diffs[k])
        if d <= 0.0:
            continue
        threshold = max(HIKE_ABS_FLOOR, 1.0 * float(roll_std[k]))
        if d <= threshold:
            continue
        left = float(diffs[k - 1]) if k - 1 >= 0 else float("-inf")
        right = float(diffs[k + 1]) if k + 1 < len(diffs) else float("-inf")
        # Local prominence: d must be >= both neighbours.
        if d < left or d < right:
            continue
        # Hike "lands" at the post-jump price, i.e. index k+1 in arr.
        hikes.append(k + 1)
    return hikes


def cycle_pos_at(t: int, hike_days: list[int]) -> int:
    """Days since the most recent hike-day with index <= t.

    Returns 0 when hike_days is empty (no signal -> treat as "just hiked").
    If t is before the first hike, returns t (days since a virtual origin).
    """
    if not hike_days:
        return 0
    last = -1
    for h in hike_days:
        if h <= t:
            last = h
        else:
            break
    if last < 0:
        return t
    return t - last


def median_cycle_len(hike_days: list[int]) -> int:
    """Median interval between consecutive hikes, banded to [4, 14].

    Returns DEFAULT_CYCLE_LEN (7) when there are fewer than 2 hikes.
    """
    if len(hike_days) < 2:
        return DEFAULT_CYCLE_LEN
    intervals = [hike_days[i + 1] - hike_days[i] for i in range(len(hike_days) - 1)]
    L = int(round(float(np.median(intervals))))
    return max(MIN_CYCLE_LEN, min(MAX_CYCLE_LEN, L))


def _tgp_return(tgp_sorted: list[tuple[date, float]], anchor_date: date, lag: int) -> float | None:
    """Recent wholesale TGP return over ``lag`` days ending at/before ``anchor_date``.

    ``tgp_sorted`` is (date, price) ascending. Returns the fractional change
    between the TGP ``lag`` days before the anchor and the TGP at/before the
    anchor — the leading-indicator signal — or None if either side is missing.
    Causal: only uses TGP on/before the anchor date.
    """
    if not tgp_sorted or lag <= 0:
        return None
    dates = [d for d, _ in tgp_sorted]
    i_now = bisect.bisect_right(dates, anchor_date) - 1
    if i_now < 0:
        return None
    i_past = bisect.bisect_right(dates, anchor_date - timedelta(days=lag)) - 1
    if i_past < 0:
        return None
    now = tgp_sorted[i_now][1]
    past = tgp_sorted[i_past][1]
    if not past:
        return None
    return now / past - 1.0


def build_row_features(
    prices: list[float],
    t: int,
    hike_days: list[int],
    cycle_len: int,
    weekday: int = 0,
) -> list[float]:
    """Build the 6 CAUSAL features for row ``t``.

    Retained as a numpy utility + the surface exercised by the causality
    regression test. The pure-fade production model does not call this (the
    fade curve replaces the GBM that consumed it), but the shape is kept so a
    future numpy ridge has a design matrix ready and the causality invariant
    stays pinned.

    Row ``t`` predicts ``prices[t]`` and uses ONLY ``prices[0..t-1]`` plus the
    pre-computed ``hike_days``. ``weekday`` is the weekday of day t (passed in
    so this function stays pure on the price list).

    Order: [cycle_pos, weekday, level, last_hike_mag, dist_to_expected_peak,
            recent_volatility]
    """
    arr = list(prices)
    # 1. cycle_pos as-of end of day t-1 (causal).
    if hike_days:
        cp = cycle_pos_at(t - 1, hike_days)
    else:
        cp = (t - 1) % cycle_len if cycle_len > 0 else 0

    # 3. level: trailing mean ending at t-1, min MIN_LEVEL_WINDOW elements.
    lo = max(0, t - TRAILING_LEN)
    level = float(np.mean(arr[lo:t])) if t > 0 else 0.0

    # 4. last_hike_mag: price jump at the most recent hike strictly before t.
    last_hike = -1
    for h in hike_days:
        if h < t:
            last_hike = h
        else:
            break
    last_hike_mag = float(arr[last_hike] - arr[last_hike - 1]) if 0 < last_hike < len(arr) else 0.0

    # 5. dist_to_expected_peak = cycle_len - cycle_pos (signed).
    dist = cycle_len - cp

    # 6. recent_volatility: rolling-7 std of day-over-day diffs ending at t-1.
    if t >= 2:
        start = max(1, t - VOLATILITY_WINDOW)
        window = [arr[i] - arr[i - 1] for i in range(start, t)]
        recent_vol = float(np.std(window)) if window else 0.0
    else:
        recent_vol = 0.0

    return [float(cp), float(weekday), level, last_hike_mag, float(dist), recent_vol]


def _fade_curve_for(
    series: list[float], hikes: list[int], L: int
) -> tuple[dict[int, float], float]:
    """Empirical fade curve over ``series``: mean price per cycle_pos % L.

    Returns (fade_curve {cp: mean_price}, fade_mean). fade_mean is the mean of
    the fade_curve values (the cycle's average level) and is the fallback for
    any cycle_pos missing from the curve.
    """
    if not hikes or L <= 0:
        return {}, float(np.mean(series)) if series else 0.0
    by_cp: dict[int, list[float]] = {}
    for i, price in enumerate(series):
        cp = cycle_pos_at(i, hikes) % L
        by_cp.setdefault(cp, []).append(price)
    fade = {cp: float(np.mean(ps)) for cp, ps in by_cp.items()}
    fade_mean = float(np.mean(list(fade.values()))) if fade else float(np.mean(series))
    return fade, fade_mean


# --- The predictor ----------------------------------------------------------
class FuelPricePredictor:
    """Cycle-aware fuel-price forecaster for one product.

    Tiered by available history (all numpy):

      - n < 7                  -> constant (repeat latest price)
      - n < 14                 -> weekday_mean (average-baseline math)
      - n >= 14 and >= 3 hikes -> fade (empirical-fade-anchored; production)
      - n >= 14, < 3 hikes     -> weekday_mean (no cycle signal to fade on)
    """

    def __init__(self) -> None:
        # Fitted-state flag (coordinator checks this).
        self._fitted: bool = False
        # Public metrics dictionary (None before fit; populated by fit).
        self.train_metrics: dict | None = None
        # Cheap fallback state (used by degraded tiers + baseline metric).
        self._weekday_mean: list[float] = [0.0] * 7
        self._overall_mean: float = 0.0
        self._recent_mean: float = 0.0
        self._latest_price: float | None = None
        self._model_kind: str = "unfitted"
        # Cycle state.
        self._hike_days: list[int] = []
        self._L: int = DEFAULT_CYCLE_LEN
        # Raw days-since-last-hike at the last training day (drives cycle_state).
        self._days_since_hike_at_fit: int = 0
        # Empirical fade curve: mean price per cycle_pos (0..L-1) across all
        # observed cycles. Drives the per-day forecast SHAPE in
        # _predict_calibrated (the cheapest-day signal). Empty when fewer than
        # one hike was detected in training -> predict falls back to weekday_mean.
        self._fade_curve: dict[int, float] = {}
        self._fade_mean: float = 0.0  # mean of fade_curve values (cycle mean)
        self._last_fit_date: date | None = None
        self._last_fit_cp: int = 0  # cycle_pos of the last training day
        # Tail context for predict() clamp.
        self._prices_tail: list[float] = []
        self._min28: float = 0.0
        self._max28: float = 0.0
        # Wholesale TGP drift term (leading indicator). β is fit on the fade
        # model's walk-forward residuals vs the TGP return over TGP_LAG_DAYS;
        # applied at predict time as a level drift. None/empty => no drift.
        self._tgp_beta: float | None = None
        self._tgp_lag: int = TGP_LAG_DAYS
        self._tgp_series: dict[date, float] = {}

    # ---- fit --------------------------------------------------------------
    def fit(
        self,
        prices_by_date: dict[date, float],
        global_history: dict[str, dict[date, float]] | None = None,  # noqa: ARG002 — accepted, ignored
        tgp_series: dict[date, float] | None = None,
    ) -> None:
        """Fit on {date: price} for one product.

        ``global_history`` is accepted for backward-compatibility of the call
        signature but is no longer used: the global leading-indicator features
        only fed the (now-removed) GBM design matrix, and the pure-fade model
        has no design matrix to extend.
        """
        if not prices_by_date:
            self._fitted = False
            return
        items = sorted(prices_by_date.items())
        dates = [d for d, _ in items]
        series = [float(p) for _, p in items]
        n = len(series)
        trained_at = datetime.now(UTC).isoformat()

        # Always populate the cheap fallback state.
        by_weekday: list[list[float]] = [[] for _ in range(7)]
        for d, p in items:
            by_weekday[d.weekday()].append(p)
        self._weekday_mean = [mean(xs) if xs else 0.0 for xs in by_weekday]
        self._overall_mean = mean(series)
        self._recent_mean = mean(series[-28:]) if n >= 1 else self._overall_mean
        self._latest_price = series[-1]
        self._prices_tail = series[-TRAILING_LEN:]
        self._min28 = min(series[-TRAILING_LEN:])
        self._max28 = max(series[-TRAILING_LEN:])

        # Tier 1: constant.
        if n < MIN_LEVEL_WINDOW:
            self._model_kind = "constant"
            self._hike_days = []
            self._L = DEFAULT_CYCLE_LEN
            self.train_metrics = self._empty_metrics(
                n_train=n,
                n_hikes=0,
                cycle_len=DEFAULT_CYCLE_LEN,
                model_kind="constant",
                trained_at=trained_at,
            )
            self._fitted = True
            return

        hikes = detect_hikes(series)
        L = median_cycle_len(hikes)
        self._hike_days = hikes
        self._L = L
        # Raw days-since-last-hike at the last training day; advanced by elapsed
        # days in cycle_state() to give the live "where are we now" position.
        self._days_since_hike_at_fit = cycle_pos_at(n - 1, hikes)

        # Empirical fade curve (the cheapest-day signal).
        self._fade_curve = {}
        self._fade_mean = self._overall_mean
        self._last_fit_date = dates[-1] if dates else None
        self._last_fit_cp = 0
        if hikes and L > 0:
            self._fade_curve, self._fade_mean = _fade_curve_for(series, hikes, L)
            self._last_fit_cp = cycle_pos_at(n - 1, hikes) % L

        # Tier 2: weekday_mean (old average-baseline math).
        if n < 14:
            self._model_kind = "weekday_mean"
            self.train_metrics = self._empty_metrics(
                n_train=n,
                n_hikes=len(hikes),
                cycle_len=L,
                model_kind="weekday_mean",
                trained_at=trained_at,
            )
            self._fitted = True
            return

        # Tier 3: full pure-fade model (needs a cycle signal — >=3 hikes).
        # Fewer hikes -> weekday_mean fallback (no cycle to fade on).
        if len(hikes) < 3:
            self._model_kind = "weekday_mean"
            self.train_metrics = self._empty_metrics(
                n_train=n,
                n_hikes=len(hikes),
                cycle_len=L,
                model_kind="weekday_mean",
                trained_at=trained_at,
            )
            self._fitted = True
            return

        self._model_kind = "fade"
        # Wholesale TGP drift term: fit β on the fade residuals vs the TGP return.
        self._tgp_series = tgp_series or {}
        tgp_sorted = sorted(self._tgp_series.items()) or None
        metrics = self._walk_forward(
            series=series, dates=dates, hold=min(28, n // 5), tgp_sorted=tgp_sorted
        )
        self._tgp_beta = metrics.get("tgp_beta")
        metrics["model_kind"] = self._model_kind
        metrics["n_train"] = n
        metrics["n_hikes"] = len(hikes)
        metrics["cycle_len_days"] = L
        metrics["trained_at"] = trained_at
        self.train_metrics = metrics
        self._fitted = True

    # ---- predict ----------------------------------------------------------
    def predict(
        self,
        start: date,
        horizon: int,
        known: dict[date, float] | None = None,
        global_recent: dict[str, dict[date, float]] | None = None,  # noqa: ARG002 — accepted, ignored
    ) -> list[DayForecast]:
        """Predict ``horizon`` days from ``start``, overriding with ``known``.

        ``global_recent`` is accepted for backward-compatibility of the call
        signature but is no longer used (the GBM+offset path that consumed it
        is gone). All forecasts use the pure-fade-anchored path.
        """
        known = known or {}
        points: list[DayForecast] = []

        # Emit known days verbatim.
        known_days: set[date] = set()
        for i in range(horizon):
            day = start + timedelta(days=i)
            if day in known:
                points.append(DayForecast(day, float(known[day]), "known"))
                known_days.add(day)
        if horizon <= len(known_days):
            return sorted(points, key=lambda p: p.day)

        # Unfitted -> None prices for forecast days.
        if not self._fitted:
            for i in range(horizon):
                day = start + timedelta(days=i)
                if day not in known_days:
                    points.append(DayForecast(day, None, "forecast"))
            return sorted(points, key=lambda p: p.day)

        # Forecast-day fill-in for the degraded tiers.
        if self._model_kind == "constant":
            self._predict_constant(start, horizon, known, known_days, points)
            return sorted(points, key=lambda p: p.day)
        if self._model_kind == "weekday_mean":
            self._predict_weekday_mean(start, horizon, known, known_days, points)
            return sorted(points, key=lambda p: p.day)

        # Full pure-fade model (the production path).
        self._predict_calibrated(start, horizon, known, known_days, points)
        return sorted(points, key=lambda p: p.day)

    # ---- cycle state (diagnostic) ----------------------------------------
    def cycle_state(self, anchor_date: date) -> dict:
        """Live cycle position as of ``anchor_date`` (today).

        Advances the fit-time days-since-last-hike by elapsed calendar days,
        mirroring exactly how ``_predict_calibrated`` advances the cycle phase
        — so this reports the cycle the model *believes* it is in, not a fresh
        re-detection on a short trailing window. Returns ``{}`` when the model
        is unfitted or was fit at a tier with no cycle signal (constant tier;
        weekday_mean/constant never set ``_last_fit_date``).

          - cycle_pos: 0..L-1 (0 = a hike just landed)
          - days_since_last_hike: raw, unmodded
          - expected_next_hike_in_days: L - cycle_pos (0 = a hike is "due")
        """
        if not self._fitted or self._last_fit_date is None or self._L <= 0:
            return {}
        elapsed = max(0, (anchor_date - self._last_fit_date).days)
        # getattr: models pickled before this attribute existed (older versions)
        # deserialize without it — degrade to "just hiked at fit" rather than raise.
        dsh = getattr(self, "_days_since_hike_at_fit", 0) + elapsed
        cp = dsh % self._L
        return {
            "cycle_pos": cp,
            "cycle_len_days": self._L,
            "days_since_last_hike": dsh,
            "expected_next_hike_in_days": max(0, self._L - cp),
        }

    # ---- cheapest (static) ------------------------------------------------
    @staticmethod
    def cheapest(points: list[DayForecast]) -> DayForecast:
        """Pick the cheapest day, preferring priced points."""
        priced = [p for p in points if p.price_cpl is not None]
        if priced:
            return min(priced, key=lambda p: p.price_cpl)
        return points[0]

    # ---- internal: degraded-tier predict ----------------------------------
    def _predict_constant(
        self,
        start: date,
        horizon: int,
        known: dict[date, float],
        known_days: set[date],
        points: list[DayForecast],
    ) -> None:
        # Repeat latest known if any; else repeat latest training price.
        if known:
            anchor_date = max(known)
            val = float(known[anchor_date])
        else:
            val = float(self._latest_price) if self._latest_price is not None else 0.0
        for i in range(horizon):
            day = start + timedelta(days=i)
            if day not in known_days:
                points.append(DayForecast(day, round(val, 1), "forecast"))

    def _predict_weekday_mean(
        self,
        start: date,
        horizon: int,
        known: dict[date, float],
        known_days: set[date],
        points: list[DayForecast],
    ) -> None:
        level = self._recent_mean or self._overall_mean
        # Optional anchor: shift the level toward the latest known price.
        anchor_shift = 0.0
        if known:
            anchor_date = max(known)
            anchor_shift = float(known[anchor_date]) - self._recent_mean
        for i in range(horizon):
            day = start + timedelta(days=i)
            if day in known_days:
                continue
            wd_mean = self._weekday_mean[day.weekday()] or self._overall_mean
            seasonal = wd_mean - self._overall_mean
            val = max(0.0, level + seasonal + anchor_shift)
            points.append(DayForecast(day, round(val, 1), "forecast"))

    # ---- internal: empirical-fade-anchored predict -----------------------
    def _predict_calibrated(
        self,
        start: date,
        horizon: int,
        known: dict[date, float],
        known_days: set[date],
        points: list[DayForecast],
    ) -> None:
        """Forecast via the empirical fade curve anchored to the known price.

        The per-day forecast SHAPE comes from the historical fade curve (mean
        price per cycle_pos), and the LEVEL is pinned by the live known anchor
        price:

            forecast(day) = anchor_price + (fade[day_cp] - fade[anchor_cp])

        The anchor's cycle_pos is derived from the FIT-TIME cycle phase
        advanced by elapsed days (mod L), NOT by re-detecting hikes on a short
        trailing window (which went out-of-distribution on real WA data and
        flattened the forecast -> argmin noise). Falls back to weekday_mean
        when no fade curve is available (too few hikes in training).
        """
        if known:
            anchor_date = max(known)
            anchor_price = float(known[anchor_date])
        else:
            anchor_date = start - timedelta(days=1)
            anchor_price = float(self._latest_price) if self._latest_price is not None else 0.0

        L = self._L
        lo_clamp = CLAMP_LO * self._min28
        hi_clamp = CLAMP_HI * self._max28

        if self._fade_curve and self._last_fit_date is not None and L > 0:
            elapsed = max(0, (anchor_date - self._last_fit_date).days)
            anchor_cp = (self._last_fit_cp + elapsed) % L
            anchor_fade = self._fade_curve.get(anchor_cp, self._fade_mean)
            # Leading-indicator level drift: β · recent wholesale TGP return.
            # getattr: models pickled before this field existed degrade to no drift.
            drift = 0.0
            _beta = getattr(self, "_tgp_beta", None)
            _tgp = getattr(self, "_tgp_series", {}) or {}
            if _beta is not None and _tgp:
                _lag = getattr(self, "_tgp_lag", TGP_LAG_DAYS)
                _gret = _tgp_return(sorted(_tgp.items()), anchor_date, _lag)
                if _gret is not None:
                    drift = _beta * _gret
            for i in range(horizon):
                day = start + timedelta(days=i)
                if day in known_days:
                    continue
                # Days from the anchor to this forecast day.
                day_cp = (self._last_fit_cp + elapsed + (day - anchor_date).days) % L
                day_fade = self._fade_curve.get(day_cp, self._fade_mean)
                final = anchor_price + (day_fade - anchor_fade) + drift
                final = max(lo_clamp, min(hi_clamp, final))
                points.append(DayForecast(day, round(final, 1), "forecast"))
            return

        # Fallback: no fade curve (too few hikes in training) -> weekday_mean.
        self._predict_weekday_mean(start, horizon, known, known_days, points)

    # ---- internal: pure-fade walk-forward holdout -------------------------
    def _walk_forward(
        self,
        series: list[float],
        dates: list[date],
        hold: int,
        tgp_sorted: list[tuple[date, float]] | None = None,
    ) -> dict:
        """Walk-forward holdout scored against the pure-fade forecast.

        For each held-out day h, train on the prefix ending at h-3 (so the
        anchor at h-2 is "known but not yet in training"), re-detect hikes on
        that prefix, rebuild the fade curve, run the pure-fade forecast
        anchored to series[h-2], and score the error vs series[h]. Also score
        the average-baseline (weekday_mean + recent_mean) on the same hold for
        comparison.
        """
        n = len(series)
        if hold < 3 or n < MIN_LEVEL_WINDOW + hold + 3:
            return {
                "mae": None,
                "mape_pct": None,
                "baseline_mae": None,
                "improvement_pct": None,
                "post_hike_mae": None,
                "normal_mae": None,
                "n_holdout": 0,
                "tgp_beta": None,
            }

        lo_clamp = CLAMP_LO * self._min28
        hi_clamp = CLAMP_HI * self._max28

        # Step so we do at most ~7 refits of the fade curve.
        step = max(1, hold // 7)
        hold_indices = list(range(n - hold, n, step))

        mae_errors: list[float] = []
        # (actual, abs_err) pairs collected UNDER THE SAME GUARD as mae_errors,
        # so each error is guaranteed to pair with its own actual. The earlier
        # MAPE rebuild from a weakly-guarded actuals list silently mispaired
        # errors with wrong-day actuals whenever a hold day was skipped.
        wf_pairs: list[tuple[float, float]] = []
        baseline_errors: list[float] = []
        post_hike_errors: list[float] = []
        normal_errors: list[float] = []
        # Signed residuals + TGP returns for the leading-indicator β fit.
        beta_resid: list[float] = []
        beta_gret: list[float] = []

        # Hike-threshold for "is h a post-hike day" labelling.
        diffs_all = np.diff(np.asarray(series, dtype=float))
        post_threshold = (
            max(HIKE_ABS_FLOOR, 1.0 * float(np.std(diffs_all)))
            if len(diffs_all)
            else HIKE_ABS_FLOOR
        )

        for h in hold_indices:
            # Need anchor at h-2 and a prefix long enough to compute a fade curve.
            anchor_t = h - 2
            prefix_end = anchor_t  # prefix = series[:anchor_t] (indices 0..h-3)
            if prefix_end < MIN_LEVEL_WINDOW + 2:
                continue
            prefix = series[:prefix_end]
            hikes_p = detect_hikes(prefix)
            L_p = median_cycle_len(hikes_p)
            # Need >=3 hikes on the prefix for a fade forecast; else skip
            # (the production tier would route this to weekday_mean).
            if len(hikes_p) < 3 or not hikes_p:
                continue
            fade_p, fade_mean_p = _fade_curve_for(prefix, hikes_p, L_p)
            if not fade_p:
                continue

            anchor_price = series[anchor_t]
            anchor_cp = cycle_pos_at(anchor_t, hikes_p) % L_p
            target_cp = cycle_pos_at(h, hikes_p) % L_p
            pred_h = anchor_price + (
                fade_p.get(target_cp, fade_mean_p) - fade_p.get(anchor_cp, fade_mean_p)
            )
            pred_h = max(lo_clamp, min(hi_clamp, pred_h))
            actual_h = series[h]
            err_h = abs(actual_h - pred_h)
            mae_errors.append(err_h)
            wf_pairs.append((actual_h, err_h))
            if tgp_sorted is not None:
                gret = _tgp_return(tgp_sorted, dates[anchor_t], TGP_LAG_DAYS)
                if gret is not None:
                    beta_resid.append(actual_h - pred_h)
                    beta_gret.append(gret)

            # Average-baseline prediction at h (weekday_mean + recent_mean),
            # trained on the same prefix.
            prefix_dates = dates[:prefix_end]
            recent = mean(prefix[-28:]) if prefix else self._overall_mean
            overall = mean(prefix) if prefix else self._overall_mean
            wd_by_day: list[list[float]] = [[] for _ in range(7)]
            for d, p in zip(prefix_dates, prefix, strict=False):
                wd_by_day[d.weekday()].append(p)
            wd_means = [mean(xs) if xs else overall for xs in wd_by_day]
            wd_h = wd_means[dates[h].weekday()]
            baseline_pred = recent + (wd_h - overall)
            baseline_errors.append(abs(actual_h - baseline_pred))

            # Post-hike flag: did day h itself see a large positive diff?
            if h >= 1 and (series[h] - series[h - 1]) > post_threshold:
                post_hike_errors.append(abs(actual_h - pred_h))
            else:
                normal_errors.append(abs(actual_h - pred_h))

        if not mae_errors:
            return {
                "mae": None,
                "mape_pct": None,
                "baseline_mae": None,
                "improvement_pct": None,
                "post_hike_mae": None,
                "normal_mae": None,
                "n_holdout": 0,
                "tgp_beta": None,
            }

        mae = float(np.mean(mae_errors))
        baseline_mae = float(np.mean(baseline_errors)) if baseline_errors else None
        # MAPE on non-zero prices: each error paired with its OWN actual,
        # collected inside the walk-forward loop under the same guard chain.
        with np.errstate(divide="ignore", invalid="ignore"):
            pct = [err / max(abs(actual), 1e-6) for actual, err in wf_pairs]
        mape_pct = float(np.mean(pct) * 100.0) if pct else None
        improvement_pct = (
            float((baseline_mae - mae) / baseline_mae * 100.0) if baseline_mae else None
        )
        post_hike_mae = float(np.mean(post_hike_errors)) if post_hike_errors else None
        normal_mae = float(np.mean(normal_errors)) if normal_errors else None
        beats = bool(mae < baseline_mae) if baseline_mae is not None else None
        if beta_gret:
            _beta_den = float(sum(g * g for g in beta_gret))
            _beta_num = sum(r * g for r, g in zip(beta_resid, beta_gret, strict=True))
            tgp_beta = (_beta_num / _beta_den) if _beta_den else None
        else:
            tgp_beta = None
        return {
            "mae": mae,
            "mape_pct": mape_pct,
            "baseline_mae": baseline_mae,
            "improvement_pct": improvement_pct,
            "post_hike_mae": post_hike_mae,
            "normal_mae": normal_mae,
            "n_holdout": len(mae_errors),
            "beats_baseline": beats,
            "tgp_beta": tgp_beta,
            "tgp_lag": TGP_LAG_DAYS,
        }

    # ---- internal: empty metrics for the bottom two tiers -----------------
    @staticmethod
    def _empty_metrics(
        n_train: int, n_hikes: int, cycle_len: int, model_kind: str, trained_at: str | None = None
    ) -> dict:
        return {
            "mae": None,
            "mape_pct": None,
            "baseline_mae": None,
            "improvement_pct": None,
            "post_hike_mae": None,
            "normal_mae": None,
            "n_train": n_train,
            "n_holdout": 0,
            "cycle_len_days": cycle_len,
            "n_hikes": n_hikes,
            "model_kind": model_kind,
            "beats_baseline": None,
            "trained_at": trained_at,
        }
