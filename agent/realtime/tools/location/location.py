import aiohttp
from pydantic import BaseModel


class Location(BaseModel):
    """Pydantic model for location data."""

    city: str
    region: str
    country: str
    latitude: float
    longitude: float


async def get_current_location() -> Location:
    """
    Determines current location via IP geolocation
    """
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("http://ipapi.co/json/") as response:
                if response.status != 200:
                    raise ValueError(
                        f"API request failed with status {response.status}"
                    )

                data = await response.json()
                return Location.model_validate(data)

    except Exception as e:
        raise ValueError(f"Location could not be determined: {e}")
