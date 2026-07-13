from datetime import date

from custom_components.fuel_predictor_wa.historic_client import (
    _parse_date,
    month_url,
    parse_csv,
    trailing_months,
)

SAMPLE = (
    "PUBLISH_DATE,TRADING_NAME,BRAND_DESCRIPTION,PRODUCT_DESCRIPTION,PRODUCT_PRICE,"
    "ADDRESS,LOCATION,POSTCODE,AREA_DESCRIPTION,REGION_DESCRIPTION\r\n"
    "01/07/2026,53 Mile Roadhouse,United,ULP,183.90,31 South Western Hwy,PINJARRA,6208,Murray,Peel\r\n"  # noqa: E501
    "01/07/2026,53 Mile Roadhouse,United,Diesel,209.90,31 South Western Hwy,PINJARRA,6208,Murray,Peel\r\n"  # noqa: E501
    "02/07/2026,Caltex,Ampol,ULP,185.00,1 Main St,BUNBURY,6230,Bunbury,South West\r\n"
)


def test_month_url_format() -> None:
    assert month_url(2026, 7) == (
        "https://warsydprdstafuelwatch.blob.core.windows.net/"
        "historical-reports/FuelWatchRetail-07-2026.csv"
    )


def test_trailing_months_wraps_year() -> None:
    assert trailing_months(date(2026, 3, 1), 5) == [
        (2026, 3), (2026, 2), (2026, 1), (2025, 12), (2025, 11)
    ]


def test_parse_date_australian() -> None:
    assert _parse_date("01/07/2026") == date(2026, 7, 1)


def test_parse_csv_filters_product_and_normalises() -> None:
    ulp = parse_csv(SAMPLE, product_description="ULP")
    assert len(ulp) == 2
    assert ulp[0] == {
        "date": date(2026, 7, 1), "price": 183.9, "product": "ULP",
        "suburb": "PINJARRA", "region": "Peel",
    }


def test_parse_csv_no_filter_returns_all() -> None:
    assert len(parse_csv(SAMPLE)) == 3
