from pydantic import BaseModel

# =============================================================================
# API Response Models (Open-Meteo API Mappings)
# =============================================================================


class OpenMeteoCurrentResponse(BaseModel):
    """Direct mapping to Open-Meteo current weather API response."""

    temperature_2m: float
    relative_humidity_2m: float
    weather_code: int
    wind_speed_10m: float
    pressure_msl: float


class OpenMeteoHourlyResponse(BaseModel):
    """Direct mapping to Open-Meteo hourly forecast API response."""

    time: list[str]
    temperature_2m: list[float]
    weather_code: list[int]
    precipitation_probability: list[float]
    wind_speed_10m: list[float]


class OpenMeteoDailyResponse(BaseModel):
    """Direct mapping to Open-Meteo daily forecast API response."""

    time: list[str]
    temperature_2m_max: list[float]
    temperature_2m_min: list[float]
    weather_code: list[int]
    precipitation_sum: list[float]
    wind_speed_10m_max: list[float]


class OpenMeteoApiResponse(BaseModel):
    """Complete Open-Meteo API response structure."""

    current: OpenMeteoCurrentResponse
    hourly: OpenMeteoHourlyResponse
    daily: OpenMeteoDailyResponse


# =============================================================================
# Domain Models (Business Logic)
# =============================================================================


class CurrentWeather(BaseModel):
    """Current weather conditions for business logic."""

    temperature: float
    humidity: float
    pressure: float
    wind_speed: float
    description: str


class HourlyForecastEntry(BaseModel):
    """Single hourly forecast entry."""

    time: str
    temperature: float
    description: str
    rain_probability: float
    wind_speed: float


class DailySummaryEntry(BaseModel):
    """Daily weather summary."""

    day: str
    max_temp: float
    min_temp: float
    description: str
    precipitation: float
    wind_max: float


class HourlyForecast(BaseModel):
    """Hourly forecasts grouped by day."""

    today: list[HourlyForecastEntry] = []
    tomorrow: list[HourlyForecastEntry] = []
    day_after_tomorrow: list[HourlyForecastEntry] = []


class WeatherData(BaseModel):
    """Complete weather data for business logic."""

    current: CurrentWeather
    hourly_forecast: HourlyForecast
    daily_summary: list[DailySummaryEntry]
