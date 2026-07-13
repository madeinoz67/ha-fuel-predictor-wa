# Fuel Predictor WA

Machine-learning fuel-price predictor for Western Australia, built on the
legislated **FuelWatch** dataset.

- Predicts the **cheapest day** to buy fuel over the next 7 days and the
  expected price (c/L).
- Lists **today's cheapest nearby stations** (price, brand, suburb, address).
- Trains on **24+ years** of FuelWatch history published by the WA Government,
  so predictions work from day one — no cold-start scrape.

Configure your **suburb**, **fuel type**, and **search radius** during setup.
Forecaster uses only `numpy` + `pandas` (lightweight, no heavy ML runtime).
