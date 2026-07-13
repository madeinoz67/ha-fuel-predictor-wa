#!/usr/bin/env python3
"""Download the bulk FuelWatch history CSV from data.wa.gov.au (CKAN).

The dataset 'FuelWatch Historic Fuel Prices' holds daily per-station prices
since Jan 2001 — used to seed the forecaster so predictions work from day one.

Run from the repo root:
    python tools/download_history.py --out data/fuelwatch_history.csv
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

# Allow running as a script (no HA runtime needed).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from custom_components.fuel_predictor_wa.const import (  # noqa: E402
    HISTORY_CKAN_BASE,
    HISTORY_DATASET_ID,
)


def package_resources() -> list[dict]:
    """List CKAN resources for the FuelWatch historic dataset."""
    url = f"{HISTORY_CKAN_BASE}/api/3/action/package_show?id={HISTORY_DATASET_ID}"
    with urllib.request.urlopen(url, timeout=60) as resp:  # noqa: S310 — public URL
        data = json.load(resp)
    return data["result"]["resources"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/fuelwatch_history.csv")
    parser.add_argument("--format", default="CSV")
    args = parser.parse_args()

    resources = package_resources()
    matches = [r for r in resources if r.get("format", "").upper() == args.format.upper()]
    if not matches:
        print(
            "No CSV resource found. Available:",
            [(r.get("name"), r.get("format")) for r in resources],
            file=sys.stderr,
        )
        sys.exit(1)

    resource = max(matches, key=lambda r: r.get("size") or 0)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {resource.get('name')} ({resource.get('size')} bytes) -> {out_path}")
    urllib.request.urlretrieve(resource["url"], out_path)  # noqa: S310 — public URL
    print("Done. Verify the CSV header against history.py column detection.")


if __name__ == "__main__":
    main()
