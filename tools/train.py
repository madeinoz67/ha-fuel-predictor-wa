#!/usr/bin/env python3
"""Train the baseline forecaster per product from the bulk history CSV -> models/.

Mirrors the predictor.fit() contract. Persists fitted parameters per product so
the live integration can load them without retraining on the HA host.

Run from the repo root:
    python tools/train.py --history data/fuelwatch_history.csv
"""

from __future__ import annotations

import argparse
import pickle
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from custom_components.fuel_predictor_wa.history import load_history  # noqa: E402
from custom_components.fuel_predictor_wa.predictor import FuelPricePredictor  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", default="data/fuelwatch_history.csv")
    parser.add_argument("--out", default="custom_components/fuel_predictor_wa/models")
    parser.add_argument("--product", type=str, default=None, help="train one product only")
    args = parser.parse_args()

    rows = load_history(args.history)
    if not rows:
        raise SystemExit("No history loaded — run download_history.py first.")

    by_product: dict[str, dict] = defaultdict(dict)
    for r in rows:
        if r["product"] is None or r["price"] is None:
            continue
        by_product[str(r["product"])][r["date"]] = r["price"]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    products = [args.product] if args.product else sorted(by_product)
    for product in products:
        series = by_product.get(product, {})
        if not series:
            print(f"skip product {product}: no data", file=sys.stderr)
            continue
        predictor = FuelPricePredictor()
        predictor.fit(series)
        artifact = out_dir / f"predictor_{product}.pkl"
        with artifact.open("wb") as fh:
            pickle.dump(
                {
                    "weekday_mean": predictor._weekday_mean,  # noqa: SLF001
                    "overall_mean": predictor._overall_mean,  # noqa: SLF001
                    "recent_mean": predictor._recent_mean,  # noqa: SLF001
                },
                fh,
            )
        print(f"trained product {product}: {len(series)} rows -> {artifact}")


if __name__ == "__main__":
    main()
