"""
Block B, tool 1: get_weather.

The agent has no real-time knowledge of current weather -- that's exactly
why this needs to be a tool rather than something answered from training
data. Open-Meteo is free with no API key or signup for non-commercial use
(confirmed against its current docs), which is why it was chosen over a
paid weather API.

Getting weather for a place name is two separate HTTP calls, not one:
1. Geocoding: turn "Kyoto" into a latitude/longitude, since the weather
   endpoint only accepts coordinates, not place names.
2. Forecast: ask for current conditions at that latitude/longitude.

No SDK exists for this API, so both calls go through plain `requests`.
"""

import requests
from langchain_core.tools import tool

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# Open-Meteo returns a numeric WMO weather code, not a text description.
# This table is copied from Open-Meteo's own docs, not guessed at.
WMO_CODES = {
    0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "fog", 48: "depositing rime fog",
    51: "light drizzle", 53: "moderate drizzle", 55: "dense drizzle",
    56: "light freezing drizzle", 57: "dense freezing drizzle",
    61: "slight rain", 63: "moderate rain", 65: "heavy rain",
    66: "light freezing rain", 67: "heavy freezing rain",
    71: "slight snowfall", 73: "moderate snowfall", 75: "heavy snowfall",
    77: "snow grains",
    80: "slight rain showers", 81: "moderate rain showers", 82: "violent rain showers",
    85: "slight snow showers", 86: "heavy snow showers",
    95: "thunderstorm", 96: "thunderstorm with slight hail",
    97: "heavy thunderstorm", 99: "thunderstorm with heavy hail",
}


@tool
def get_weather(location: str) -> str:
    """Get the current weather for a place. `location` should be a place
    name, like 'Kyoto' or 'Kyoto, Japan' -- add a country if the name is
    ambiguous. Returns temperature, conditions, humidity and wind."""

    try:
        geo_resp = requests.get(
            GEOCODING_URL,
            params={"name": location, "count": 1},
            timeout=10,
        )
        geo_resp.raise_for_status()
        geo_results = geo_resp.json().get("results")
    except requests.RequestException as e:
        return f"Weather lookup failed while finding '{location}': {e}"

    if not geo_results:
        return f"Couldn't find a place called '{location}'. Try being more specific, e.g. adding the country."

    place = geo_results[0]
    lat, lon = place["latitude"], place["longitude"]
    resolved_name = f"{place['name']}, {place.get('country', '')}".strip(", ")

    try:
        forecast_resp = requests.get(
            FORECAST_URL,
            params={
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_2m,weather_code,wind_speed_10m,relative_humidity_2m",
                "timezone": "auto",
            },
            timeout=10,
        )
        forecast_resp.raise_for_status()
        current = forecast_resp.json()["current"]
    except requests.RequestException as e:
        return f"Weather lookup failed while getting the forecast for {resolved_name}: {e}"

    condition = WMO_CODES.get(
        current["weather_code"], f"unknown conditions (code {current['weather_code']})"
    )

    return (
        f"Current weather in {resolved_name}: {condition}, "
        f"{current['temperature_2m']}°C, "
        f"{current['relative_humidity_2m']}% humidity, "
        f"wind {current['wind_speed_10m']} km/h."
    )