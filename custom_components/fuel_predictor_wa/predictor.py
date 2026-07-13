"""numpy/pandas-free baseline fuel-price forecaster.

Seasonal baseline: per-product weekday mean + recent level. Deliberately
lightweight (no sklearn/onnx) to keep HA requirements minimal. The
fit/predict contract is stable so a stronger model can replace the
internals later (see tools/train.py).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, timedelta
from statistics import mean

_LOGGER = logging.getLogger(__name__)


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


class FuelPricePredictor:
    """Seasonal baseline forecaster for one product."""

    def __init__(self) -> None:
        self._weekday_mean: list[float] = [0.0] * 7
        self._overall_mean: float = 0.0
        self._recent_mean: float = 0.0
        self._fitted: bool = False

    def fit(self, prices_by_date: dict[date, float]) -> None:
        """Fit on {date: price} for one product."""
        if not prices_by_date:
            self._fitted = False
            return
        by_weekday: list[list[float]] = [[] for _ in range(7)]
        for d, price in prices_by_date.items():
            by_weekday[d.weekday()].append(price)
        self._weekday_mean = [mean(xs) if xs else 0.0 for xs in by_weekday]
        self._overall_mean = mean(prices_by_date.values())
        recent = sorted(prices_by_date.items())[-28:]
        self._recent_mean = mean(p for _, p in recent) if recent else self._overall_mean
        self._fitted = True

    def predict(
        self,
        start: date,
        horizon: int,
        known: dict[date, float] | None = None,
    ) -> list[DayForecast]:
        """Predict `horizon` days from `start`, overriding with `known` prices."""
        known = known or {}
        points: list[DayForecast] = []
        for i in range(horizon):
            day = start + timedelta(days=i)
            if day in known:
                points.append(DayForecast(day, float(known[day]), "known"))
            elif self._fitted:
                level = self._recent_mean or self._overall_mean
                wd_mean = self._weekday_mean[day.weekday()] or self._overall_mean
                seasonal = wd_mean - self._overall_mean
                points.append(DayForecast(day, max(0.0, level + seasonal), "forecast"))
            else:
                points.append(DayForecast(day, None, "forecast"))
        return points

    @staticmethod
    def cheapest(points: list[DayForecast]) -> DayForecast:
        """Pick the cheapest day, preferring priced points."""
        priced = [p for p in points if p.price_cpl is not None]
        if priced:
            return min(priced, key=lambda p: p.price_cpl)
        return points[0]
