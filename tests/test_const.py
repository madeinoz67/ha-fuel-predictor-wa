from custom_components.fuel_predictor_wa import const
from custom_components.fuel_predictor_wa.const import (
    PRODUCT_DIESEL,
    PRODUCT_P98,
    PRODUCT_UNLEADED,
    PRODUCTS,
)


def test_every_configurable_product_has_a_csv_description() -> None:
    for code in PRODUCTS:
        assert code in const.PRODUCT_CSV_DESCRIPTION, f"missing CSV description for product {code}"


def test_known_csv_descriptions() -> None:
    assert const.PRODUCT_CSV_DESCRIPTION[PRODUCT_UNLEADED] == "ULP"
    assert const.PRODUCT_CSV_DESCRIPTION[PRODUCT_P98] == "98 RON"
    assert const.PRODUCT_CSV_DESCRIPTION[PRODUCT_DIESEL] == "Diesel"


def test_tuning_and_status_constants() -> None:
    assert const.HISTORY_MONTHS_TARGET >= 12
    assert const.MIN_MONTHS_TO_TRAIN >= 1
    assert const.STATUS_READY == "ready"
