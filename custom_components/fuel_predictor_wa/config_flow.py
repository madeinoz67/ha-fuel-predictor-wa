"""Config flow for Fuel Predictor WA."""
from __future__ import annotations

from typing import Any

import voluptuous as vol
from homeassistant import config_entries
from homeassistant.core import callback

from .const import (
    CONF_FORECAST_HORIZON_DAYS,
    CONF_PRODUCT,
    CONF_RADIUS_KM,
    CONF_STATION_LIMIT,
    CONF_SUBURB,
    CONF_SURROUNDING,
    DEFAULT_FORECAST_HORIZON_DAYS,
    DEFAULT_PRODUCT,
    DEFAULT_RADIUS_KM,
    DEFAULT_STATION_LIMIT,
    DEFAULT_SURROUNDING,
    DOMAIN,
    PRODUCTS,
)

STEP_USER_DATA_SCHEMA = vol.Schema(
    {
        vol.Required(CONF_PRODUCT, default=DEFAULT_PRODUCT): vol.In(PRODUCTS),
        vol.Required(CONF_SUBURB): str,
        vol.Optional(CONF_SURROUNDING, default=DEFAULT_SURROUNDING): bool,
        vol.Optional(CONF_RADIUS_KM, default=DEFAULT_RADIUS_KM): vol.All(
            vol.Coerce(int), vol.Range(min=1, max=100)
        ),
        vol.Optional(
            CONF_FORECAST_HORIZON_DAYS, default=DEFAULT_FORECAST_HORIZON_DAYS
        ): vol.All(vol.Coerce(int), vol.Range(min=2, max=14)),
        vol.Optional(CONF_STATION_LIMIT, default=DEFAULT_STATION_LIMIT): vol.All(
            vol.Coerce(int), vol.Range(min=1, max=20)
        ),
    }
)


class FuelPredictorConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handle a config flow for Fuel Predictor WA."""

    VERSION = 1

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> config_entries.FlowResult:
        """Handle the initial step."""
        errors: dict[str, str] = {}
        if user_input is not None:
            await self.async_set_unique_id(
                f"{DOMAIN}_{user_input[CONF_PRODUCT]}_{user_input[CONF_SUBURB].lower()}"
            )
            self._abort_if_unique_id_configured()
            return self.async_create_entry(
                title=f"{PRODUCTS[user_input[CONF_PRODUCT]]} — {user_input[CONF_SUBURB]}",
                data=user_input,
            )
        return self.async_show_form(
            step_id="user", data_schema=STEP_USER_DATA_SCHEMA, errors=errors
        )

    @staticmethod
    @callback
    def async_get_options_flow(
        config_entry: config_entries.ConfigEntry,
    ) -> config_entries.OptionsFlow:
        """Get the options flow."""
        return FuelPredictorOptionsFlow(config_entry)


class FuelPredictorOptionsFlow(config_entries.OptionsFlow):
    """Options flow for adjusting search config."""

    def __init__(self, config_entry: config_entries.ConfigEntry) -> None:
        self.config_entry = config_entry

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> config_entries.FlowResult:
        if user_input is not None:
            return self.async_create_entry(title="", data=user_input)
        data = self.config_entry.data
        schema = vol.Schema(
            {
                vol.Required(
                    CONF_PRODUCT, default=data.get(CONF_PRODUCT, DEFAULT_PRODUCT)
                ): vol.In(PRODUCTS),
                vol.Required(CONF_SUBURB, default=data.get(CONF_SUBURB, "")): str,
                vol.Optional(
                    CONF_SURROUNDING, default=data.get(CONF_SURROUNDING, DEFAULT_SURROUNDING)
                ): bool,
                vol.Optional(
                    CONF_RADIUS_KM, default=data.get(CONF_RADIUS_KM, DEFAULT_RADIUS_KM)
                ): vol.All(vol.Coerce(int), vol.Range(min=1, max=100)),
                vol.Optional(
                    CONF_FORECAST_HORIZON_DAYS,
                    default=data.get(CONF_FORECAST_HORIZON_DAYS, DEFAULT_FORECAST_HORIZON_DAYS),
                ): vol.All(vol.Coerce(int), vol.Range(min=2, max=14)),
            }
        )
        return self.async_show_form(step_id="init", data_schema=schema)
