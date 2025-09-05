from datetime import datetime, timedelta
from textwrap import dedent

from agent.realtime.tools.location.location import Location
from agent.realtime.tools.weather.views import (
    WeatherData,
    CurrentWeather,
    DailySummaryEntry,
    HourlyForecastEntry,
    OpenMeteoCurrentResponse,
    OpenMeteoHourlyResponse,
    OpenMeteoDailyResponse,
    OpenMeteoApiResponse,
    HourlyForecast,
)


# =============================================================================
# Constants
# =============================================================================

_WEATHER_CODE_DESCRIPTIONS = {
    # Clear conditions
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    # Fog conditions
    45: "Fog",
    48: "Depositing rime fog",
    # Drizzle conditions
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    56: "Light freezing drizzle",
    57: "Dense freezing drizzle",
    # Rain conditions
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    66: "Light freezing rain",
    67: "Heavy freezing rain",
    # Snow conditions
    71: "Slight snowfall",
    73: "Moderate snowfall",
    75: "Heavy snowfall",
    77: "Snow grains",
    # Shower conditions
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    # Thunderstorm conditions
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}

_DAY_NAMES = ["Today", "Tomorrow", "Day after tomorrow"]


def transform_api_response(api_response: OpenMeteoApiResponse) -> WeatherData:
    """Transform complete API response to domain model."""
    current = _transform_current_weather(api_response.current)
    hourly_forecast = _transform_hourly_forecast(api_response.hourly)
    daily_summary = _transform_daily_summary(api_response.daily)

    return WeatherData(
        current=current, hourly_forecast=hourly_forecast, daily_summary=daily_summary
    )


def format_weather_report(weather_data: WeatherData, location: Location) -> str:
    """Format complete weather data into a readable report."""
    # Build all sections
    current_section = _format_current_weather_section(weather_data.current, location)
    daily_overview = _format_daily_overview_section(weather_data.daily_summary)
    hourly_sections = _get_forecast_sections(weather_data.hourly_forecast)
    footer = _format_footer()

    # Combine all sections
    return current_section + daily_overview + "".join(hourly_sections) + footer


# =============================================================================
# Helper Functions
# =============================================================================


def _get_weather_description(weather_code: int) -> str:
    """Get weather description from weather code."""
    return _WEATHER_CODE_DESCRIPTIONS.get(weather_code, "Unknown")


def _safe_get_list_item(items: list, index: int, default=0):
    """Safely get item from list with fallback to default."""
    return items[index] if index < len(items) else default


def _get_date_categories():
    """Get today, tomorrow, and day after tomorrow dates."""
    now = datetime.now()
    today = now.date()
    tomorrow = today + timedelta(days=1)
    day_after_tomorrow = today + timedelta(days=2)
    return today, tomorrow, day_after_tomorrow


def _categorize_hourly_entry_by_date(dt_date, today, tomorrow, day_after_tomorrow):
    """Categorize date into today/tomorrow/day_after_tomorrow."""
    if dt_date == today:
        return "today"
    elif dt_date == tomorrow:
        return "tomorrow"
    elif dt_date == day_after_tomorrow:
        return "day_after_tomorrow"
    return None


# =============================================================================
# Transformation Functions (Private)
# =============================================================================


def _transform_current_weather(api_current: OpenMeteoCurrentResponse) -> CurrentWeather:
    """Transform API current weather to domain model."""
    return CurrentWeather(
        temperature=api_current.temperature_2m,
        humidity=api_current.relative_humidity_2m,
        pressure=api_current.pressure_msl,
        wind_speed=api_current.wind_speed_10m,
        description=_get_weather_description(api_current.weather_code),
    )


def _create_hourly_entry(
    api_hourly: OpenMeteoHourlyResponse, index: int, dt: datetime
) -> HourlyForecastEntry:
    """Create a single hourly forecast entry."""
    return HourlyForecastEntry(
        time=dt.strftime("%H:%M"),
        temperature=_safe_get_list_item(api_hourly.temperature_2m, index),
        description=_get_weather_description(
            _safe_get_list_item(api_hourly.weather_code, index)
        ),
        rain_probability=_safe_get_list_item(
            api_hourly.precipitation_probability, index
        ),
        wind_speed=_safe_get_list_item(api_hourly.wind_speed_10m, index),
    )


def _transform_hourly_forecast(api_hourly: OpenMeteoHourlyResponse) -> HourlyForecast:
    """Transform API hourly data to domain model."""
    today, tomorrow, day_after_tomorrow = _get_date_categories()
    forecast_data = {"today": [], "tomorrow": [], "day_after_tomorrow": []}

    # Process every 3rd hour
    for i in range(0, len(api_hourly.time), 3):
        if i >= len(api_hourly.time):
            continue

        time_str = api_hourly.time[i]
        dt = datetime.fromisoformat(time_str.replace("Z", "+00:00"))

        category = _categorize_hourly_entry_by_date(
            dt.date(), today, tomorrow, day_after_tomorrow
        )

        if category:
            entry = _create_hourly_entry(api_hourly, i, dt)
            forecast_data[category].append(entry)

    return HourlyForecast(**forecast_data)


def _create_daily_entry(
    api_daily: OpenMeteoDailyResponse, index: int, day_name: str
) -> DailySummaryEntry:
    """Create a single daily summary entry."""
    return DailySummaryEntry(
        day=day_name,
        max_temp=_safe_get_list_item(api_daily.temperature_2m_max, index),
        min_temp=_safe_get_list_item(api_daily.temperature_2m_min, index),
        description=_get_weather_description(
            _safe_get_list_item(api_daily.weather_code, index)
        ),
        precipitation=_safe_get_list_item(api_daily.precipitation_sum, index),
        wind_max=_safe_get_list_item(api_daily.wind_speed_10m_max, index),
    )


def _transform_daily_summary(
    api_daily: OpenMeteoDailyResponse,
) -> list[DailySummaryEntry]:
    """Transform API daily data to domain model."""
    daily_entries = []
    max_days = min(len(api_daily.time), len(_DAY_NAMES))

    for i in range(max_days):
        entry = _create_daily_entry(api_daily, i, _DAY_NAMES[i])
        daily_entries.append(entry)

    return daily_entries


# =============================================================================
# Report Formatting Functions (Private)
# =============================================================================


def _format_current_weather_section(current: CurrentWeather, location: Location) -> str:
    """Format current weather section."""
    return dedent(
        f"""
        **Weather Report for {location.city}, {location.country}**

        **Location:** {location.city} ({location.latitude:.2f}, {location.longitude:.2f})

        **Current ({datetime.now().strftime("%H:%M")}):**
           • Temperature: {current.temperature:.1f}°C
           • Conditions: {current.description}
           • Humidity: {current.humidity:.0f}%
           • Wind: {current.wind_speed:.1f} m/s
           • Pressure: {current.pressure:.0f} hPa
    """
    ).strip()


def _format_daily_summary_line(day_data: DailySummaryEntry) -> str:
    """Format a single daily summary line."""
    return (
        f"    **{day_data.day}:** {day_data.min_temp:.1f}°C - {day_data.max_temp:.1f}°C • "
        f"{day_data.description} • Precipitation: {day_data.precipitation:.1f}mm • "
        f"Wind max: {day_data.wind_max:.1f}m/s"
    )


def _format_daily_overview_section(daily_summary: list[DailySummaryEntry]) -> str:
    """Format daily overview section."""
    section = "\n\n**3-Day Overview:**"

    for day_data in daily_summary:
        section += f"\n{_format_daily_summary_line(day_data)}"

    return section


def _format_hourly_entry_line(entry: HourlyForecastEntry) -> str:
    """Format a single hourly forecast line."""
    return (
        f"   🕐 {entry.time}: 🌡️ {entry.temperature:.1f}°C • {entry.description} • "
        f"Rain possibility: {entry.rain_probability:.0f}% • Wind: {entry.wind_speed:.1f}m/s"
    )


def _format_hourly_forecast_section(
    day_name: str, entries: list[HourlyForecastEntry]
) -> str:
    """Format hourly forecast section for a specific day."""
    if not entries:
        return ""

    section = f"\n\n    **{day_name} - Hourly Forecast:**"

    for entry in entries:
        section += f"\n{_format_hourly_entry_line(entry)}"

    return section


def _get_forecast_sections(hourly_forecast: HourlyForecast) -> list[str]:
    """Get all hourly forecast sections."""
    forecast_days = [
        ("Today", hourly_forecast.today),
        ("Tomorrow", hourly_forecast.tomorrow),
        ("Day after tomorrow", hourly_forecast.day_after_tomorrow),
    ]

    sections = []
    for day_name, entries in forecast_days:
        section = _format_hourly_forecast_section(day_name, entries)
        if section:
            sections.append(section)

    return sections


def _format_footer() -> str:
    """Format report footer with timestamp."""
    return f"\n\n**Last updated:** {datetime.now().strftime('%H:%M:%S')}"
