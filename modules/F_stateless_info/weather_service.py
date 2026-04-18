from __future__ import annotations

from datetime import datetime, timedelta

import requests

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
_TIMEOUT = 15

_WEATHER_CODE_LABELS = {
    0: "clear sky",
    1: "mainly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "fog",
    48: "depositing rime fog",
    51: "light drizzle",
    53: "moderate drizzle",
    55: "dense drizzle",
    56: "light freezing drizzle",
    57: "dense freezing drizzle",
    61: "slight rain",
    63: "moderate rain",
    65: "heavy rain",
    66: "light freezing rain",
    67: "heavy freezing rain",
    71: "slight snow fall",
    73: "moderate snow fall",
    75: "heavy snow fall",
    77: "snow grains",
    80: "slight rain showers",
    81: "moderate rain showers",
    82: "violent rain showers",
    85: "slight snow showers",
    86: "heavy snow showers",
    95: "thunderstorm",
    96: "thunderstorm with slight hail",
    99: "thunderstorm with heavy hail",
}


def _resolve_date(day: str | None) -> str:
    today = datetime.now().date()

    if not day or day == "today":
        return today.isoformat()
    if day == "tomorrow":
        return (today + timedelta(days=1)).isoformat()

    return day


def get_weather(city: str, day: str | None = "today") -> dict:
    city = (city or "").strip()
    if not city:
        return {"success": False, "error": "No city provided."}

    geo = requests.get(
        GEOCODE_URL,
        params={"name": city, "count": 1, "language": "en", "format": "json"},
        timeout=_TIMEOUT,
    )
    geo.raise_for_status()
    geo_json = geo.json()
    results = geo_json.get("results") or []

    if not results:
        return {"success": False, "error": f"I could not find '{city}'."}

    place = results[0]
    lat = place["latitude"]
    lon = place["longitude"]
    resolved_date = _resolve_date(day)

    forecast = requests.get(
        FORECAST_URL,
        params={
            "latitude": lat,
            "longitude": lon,
            "timezone": "auto",
            "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max,precipitation_sum",
            "forecast_days": 7,
        },
        timeout=_TIMEOUT,
    )
    forecast.raise_for_status()
    data = forecast.json().get("daily", {})
    days = data.get("time") or []

    if resolved_date not in days:
        return {"success": False, "error": f"No forecast is available for {resolved_date}."}

    idx = days.index(resolved_date)
    weather_code = (data.get("weather_code") or [None])[idx]

    return {
        "success": True,
        "city": place.get("name", city),
        "country": place.get("country", ""),
        "admin1": place.get("admin1", ""),
        "date": resolved_date,
        "summary": _WEATHER_CODE_LABELS.get(weather_code, "unknown conditions"),
        "temp_max_c": (data.get("temperature_2m_max") or [None])[idx],
        "temp_min_c": (data.get("temperature_2m_min") or [None])[idx],
        "precip_probability_max": (data.get("precipitation_probability_max") or [None])[idx],
        "precipitation_sum_mm": (data.get("precipitation_sum") or [None])[idx],
        "weather_code": weather_code,
        "provider": "Open-Meteo",
    }


def weather_to_text(result: dict) -> str:
    location_bits = [result.get("city"), result.get("admin1"), result.get("country")]
    location = ", ".join([x for x in location_bits if x])
    date = result.get("date", "today")
    summary = result.get("summary", "unknown conditions")
    high = result.get("temp_max_c")
    low = result.get("temp_min_c")
    rain_prob = result.get("precip_probability_max")
    rain_mm = result.get("precipitation_sum_mm")

    return (
        f"Weather for {location} on {date}: {summary}. "
        f"High {high}°C, low {low}°C, precipitation probability up to {rain_prob}%, "
        f"expected precipitation {rain_mm} mm."
    )