"""Config-flow regression tests.

OptionsFlow.config_entry is a read-only property injected by Home Assistant.
The flow must construct with NO arguments (regression for the config-flow 500
caused by assigning self.config_entry in __init__).
"""

from __future__ import annotations

from custom_components.fuel_predictor_wa.config_flow import (
    FuelPredictorConfigFlow,
    FuelPredictorOptionsFlow,
)


def test_options_flow_constructs_without_args() -> None:
    """HA injects config_entry; the constructor takes no args."""
    FuelPredictorOptionsFlow()


def test_config_flow_constructs() -> None:
    FuelPredictorConfigFlow()
