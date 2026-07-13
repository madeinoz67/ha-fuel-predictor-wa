"""Cycle-aware HGBR + offset-calibration fuel-price forecaster.

Replaces the average-baseline internals (which systematically under-forecast
post-hike by ~the recent_mean-to-peak gap, ~19c/L on WA's weekly cycle) while
preserving the public contract the coordinator depends on:

  - class FuelPricePredictor with fit / predict / cheapest
  - DayForecast(day, price_cpl, source)
  - ForecastResult(points, cheapest_day) with .cheapest_price
  - the _fitted flag

Design:

  1. detect_hikes() finds prominent positive price spikes (cycle starts).
  2. build_row_features() turns each day into 6 CAUSAL features (only past
     prices) — cycle_pos, weekday, level, last_hike_mag, dist_to_expected_peak,
     recent_volatility.
  3. fit() picks a tier by data volume (constant < weekday_mean < ridge_degraded
     < histgbr), measures itself on a walk-forward holdout, then refits on all
     data for production.
  4. predict() applies OFFSET CALIBRATION: the difference between the model's
     raw prediction at a known anchor day and that day's actual price becomes a
     level-shifting offset added to every forecast day. This pins the forecast
     to the live known price and is what kills the post-hike bias.

sklearn is imported lazily inside fit() so the unfitted fast path (and the
tests that don't need it) never require sklearn on the import graph.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, timedelta
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


def build_row_features(
    prices: list[float],
    t: int,
    hike_days: list[int],
    cycle_len: int,
    weekday: int = 0,
) -> list[float]:
    """Build the 6 CAUSAL features for row ``t``.

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


# --- The predictor ----------------------------------------------------------
class FuelPricePredictor:
    """Cycle-aware fuel-price forecaster for one product.

    Tiered by available history:

      - n < 7            -> constant (repeat latest price)
      - n < 14           -> weekday_mean (old average-baseline math)
      - n < 35 or <3 hikes -> ridge_degraded (Ridge on weekday + level)
      - else             -> histgbr (HistGradientBoostingRegressor on 6 features)

    Every tier with a regressor uses offset calibration at predict time.
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
        # Regressor + feature config (set by fit when a regressor is chosen).
        self._regressor = None  # sklearn estimator (Ridge or HGBR)
        self._feature_cols: list[int] = list(range(6))
        self._model_kind: str = "unfitted"
        # Cycle state.
        self._hike_days: list[int] = []
        self._L: int = DEFAULT_CYCLE_LEN
        # Tail context for predict().
        self._prices_tail: list[float] = []
        self._min28: float = 0.0
        self._max28: float = 0.0

    # ---- fit --------------------------------------------------------------
    def fit(self, prices_by_date: dict[date, float]) -> None:
        """Fit on {date: price} for one product."""
        if not prices_by_date:
            self._fitted = False
            return
        items = sorted(prices_by_date.items())
        dates = [d for d, _ in items]
        series = [float(p) for _, p in items]
        n = len(series)

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
                n_train=n, n_hikes=0, cycle_len=DEFAULT_CYCLE_LEN, model_kind="constant"
            )
            self._fitted = True
            return

        hikes = detect_hikes(series)
        L = median_cycle_len(hikes)
        self._hike_days = hikes
        self._L = L

        # Tier 2: weekday_mean (old average-baseline math).
        if n < 14:
            self._model_kind = "weekday_mean"
            self.train_metrics = self._empty_metrics(
                n_train=n, n_hikes=len(hikes), cycle_len=L, model_kind="weekday_mean"
            )
            self._fitted = True
            return

        # Build the full feature matrix (rows where features are defined).
        rows_t = list(range(MIN_LEVEL_WINDOW, n))
        X = np.asarray(
            [build_row_features(series, t, hikes, L, weekday=dates[t].weekday()) for t in rows_t],
            dtype=float,
        )
        y = np.asarray([series[t] for t in rows_t], dtype=float)

        # Tier selection for the regressor.
        if n < 35 or len(hikes) < 3:
            self._model_kind = "ridge_degraded"
            from sklearn.linear_model import Ridge

            def _factory() -> Ridge:
                return Ridge(alpha=1.0)

            feature_cols = [1, 2]  # weekday, level
        else:
            self._model_kind = "histgbr"
            from sklearn.ensemble import HistGradientBoostingRegressor

            def _factory() -> HistGradientBoostingRegressor:  # type: ignore[override]
                return HistGradientBoostingRegressor(
                    loss="squared_error",
                    max_iter=300,
                    learning_rate=0.05,
                    max_leaf_nodes=15,
                    min_samples_leaf=20,
                    l2_regularization=1.0,
                    early_stopping=True,
                    random_state=42,
                    validation_fraction=0.15,
                )

            feature_cols = list(range(6))

        # Walk-forward holdout (honest fit metric). Step so we do at most
        # ~7 refits even on long series -> stays well under 1.5s on ~730 rows.
        hold = min(28, n // 5)
        metrics = self._walk_forward(
            series=series,
            dates=dates,
            rows_t=rows_t,
            X=X,
            y=y,
            feature_cols=feature_cols,
            factory=_factory,
            hold=hold,
            hikes=hikes,
        )

        # Refit the chosen regressor on ALL data for production.
        regressor = _factory()
        regressor.fit(X[:, feature_cols], y)
        self._regressor = regressor
        self._feature_cols = feature_cols

        metrics["model_kind"] = self._model_kind
        metrics["n_train"] = n
        metrics["n_hikes"] = len(hikes)
        metrics["cycle_len_days"] = L
        self.train_metrics = metrics
        self._fitted = True

    # ---- predict ----------------------------------------------------------
    def predict(
        self,
        start: date,
        horizon: int,
        known: dict[date, float] | None = None,
    ) -> list[DayForecast]:
        """Predict ``horizon`` days from ``start``, overriding with ``known``."""
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

        # Full regressor path with offset calibration (ridge_degraded + histgbr).
        self._predict_calibrated(start, horizon, known, known_days, points)
        return sorted(points, key=lambda p: p.day)

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

    # ---- internal: offset-calibrated predict ------------------------------
    def _predict_calibrated(
        self,
        start: date,
        horizon: int,
        known: dict[date, float],
        known_days: set[date],
        points: list[DayForecast],
    ) -> None:
        # Build trailing context = self._prices_tail + sorted known prices.
        trailing: list[float] = list(self._prices_tail)
        known_sorted = sorted(known.items())
        for _d, p in known_sorted:
            trailing.append(float(p))

        # Detect hikes on the initial trailing (uses ONLY known prices, never
        # forecast outputs). Forecast outputs appended below do NOT refresh
        # hike_days.
        hike_days = detect_hikes(trailing) if trailing else list(self._hike_days)
        L = self._L

        # Anchor = latest known day. If no known, use the last element of
        # trailing as a virtual anchor at start - 1 day.
        if known:
            anchor_date = max(known)
            anchor_price = float(known[anchor_date])
        else:
            anchor_date = start - timedelta(days=1)
            anchor_price = trailing[-1] if trailing else 0.0

        # Map the anchor to a row index in feature space. The anchor is the
        # last element of trailing (its row index is len(trailing)-1).
        anchor_t = max(MIN_LEVEL_WINDOW, len(trailing) - 1)
        anchor_features = build_row_features(
            trailing, anchor_t, hike_days, L, weekday=anchor_date.weekday()
        )
        raw_anchor = float(
            self._regressor.predict(
                np.asarray([anchor_features], dtype=float)[:, self._feature_cols]
            )[0]
        )
        # THE LOAD-BEARING LINE: offset pins the level to the live anchor price.
        offset = anchor_price - raw_anchor

        lo_clamp = CLAMP_LO * self._min28
        hi_clamp = CLAMP_HI * self._max28

        for i in range(horizon):
            day = start + timedelta(days=i)
            if day in known_days:
                continue
            t = len(trailing)  # next index after current trailing
            feats = build_row_features(trailing, t, hike_days, L, weekday=day.weekday())
            raw = float(
                self._regressor.predict(np.asarray([feats], dtype=float)[:, self._feature_cols])[0]
            )
            final = raw + offset
            final = max(lo_clamp, min(hi_clamp, final))
            final = round(final, 1)
            points.append(DayForecast(day, final, "forecast"))
            # Recursive feed-back so lag/level advance.
            trailing.append(final)

    # ---- internal: walk-forward holdout -----------------------------------
    def _walk_forward(
        self,
        series: list[float],
        dates: list[date],
        rows_t: list[int],
        X: np.ndarray,
        y: np.ndarray,
        feature_cols: list[int],
        factory,
        hold: int,
        hikes: list[int],
    ) -> dict:
        """Walk-forward holdout. For each held-out day h, fit on the prefix
        ending before h-2 (anchor = day h-2), forecast h with offset, score.
        Also score the average-baseline (weekday_mean + recent_mean) on the
        same hold for comparison.
        """
        n = len(series)
        if hold < 3 or len(rows_t) < MIN_LEVEL_WINDOW + hold:
            # Too little data for an honest holdout -> metrics None.
            return {
                "mae": None,
                "mape_pct": None,
                "baseline_mae": None,
                "improvement_pct": None,
                "post_hike_mae": None,
                "normal_mae": None,
                "n_holdout": 0,
            }

        t_to_row = {t: i for i, t in enumerate(rows_t)}
        lo_clamp = CLAMP_LO * self._min28
        hi_clamp = CLAMP_HI * self._max28

        # Step so we do at most ~7 refits.
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

        # Hike-threshold for "is h a post-hike day" labelling.
        diffs_all = np.diff(np.asarray(series, dtype=float))
        post_threshold = (
            max(HIKE_ABS_FLOOR, 1.0 * float(np.std(diffs_all)))
            if len(diffs_all)
            else HIKE_ABS_FLOOR
        )

        for h in hold_indices:
            # Need features(h) and features(h-2) computable.
            if h < MIN_LEVEL_WINDOW + 2 or (h - 2) not in t_to_row or h not in t_to_row:
                continue
            anchor_t = h - 2
            # Train rows: rows_t < anchor_t (features at those rows see only
            # prices[0..t-1] with t-1 <= anchor_t-1 = h-3; never the anchor).
            train_idx = [i for i, t in enumerate(rows_t) if t < anchor_t]
            if len(train_idx) < MIN_LEVEL_WINDOW:
                continue
            X_train = X[train_idx][:, feature_cols]
            y_train = y[train_idx]
            try:
                reg = factory()
                reg.fit(X_train, y_train)
            except Exception:  # noqa: BLE001 — WF must not abort fit
                continue

            anchor_row = t_to_row[anchor_t]
            h_row = t_to_row[h]
            raw_anchor = float(reg.predict(X[anchor_row : anchor_row + 1][:, feature_cols])[0])
            raw_h = float(reg.predict(X[h_row : h_row + 1][:, feature_cols])[0])
            anchor_price = series[anchor_t]
            offset = anchor_price - raw_anchor
            pred_h = max(lo_clamp, min(hi_clamp, raw_h + offset))
            actual_h = series[h]
            err_h = abs(actual_h - pred_h)
            mae_errors.append(err_h)
            wf_pairs.append((actual_h, err_h))

            # Average-baseline prediction at h (the OLD predictor's math),
            # trained on the same prefix.
            prefix_prices = series[: h - 1]
            prefix_dates = dates[: h - 1]
            recent = mean(prefix_prices[-28:]) if prefix_prices else self._overall_mean
            overall = mean(prefix_prices) if prefix_prices else self._overall_mean
            wd_by_day: list[list[float]] = [[] for _ in range(7)]
            for d, p in zip(prefix_dates, prefix_prices, strict=False):
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
            }

        mae = float(np.mean(mae_errors))
        baseline_mae = float(np.mean(baseline_errors)) if baseline_errors else None
        # MAPE on non-zero prices: each error paired with its OWN actual,
        # collected inside the walk-forward loop under the same guard chain.
        with np.errstate(divide="ignore", invalid="ignore"):
            pct = [err / max(abs(actual), 1e-6) for actual, err in wf_pairs]
        mape_pct = float(np.mean(pct) * 100.0) if pct else None
        improvement_pct = float((baseline_mae - mae) / baseline_mae) if baseline_mae else None
        post_hike_mae = float(np.mean(post_hike_errors)) if post_hike_errors else None
        normal_mae = float(np.mean(normal_errors)) if normal_errors else None
        beats = bool(mae < baseline_mae) if baseline_mae is not None else None
        return {
            "mae": mae,
            "mape_pct": mape_pct,
            "baseline_mae": baseline_mae,
            "improvement_pct": improvement_pct,
            "post_hike_mae": post_hike_mae,
            "normal_mae": normal_mae,
            "n_holdout": len(mae_errors),
            "beats_baseline": beats,
        }

    # ---- internal: empty metrics for the bottom two tiers -----------------
    @staticmethod
    def _empty_metrics(n_train: int, n_hikes: int, cycle_len: int, model_kind: str) -> dict:
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
        }
