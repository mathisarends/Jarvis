from pydantic import BaseModel, field_validator


class ResponseSpeedAdjustment(BaseModel):
    new_response_speed: float

    @field_validator("new_response_speed")
    @classmethod
    def clamp_speed(cls, v: float) -> float:
        MIN_RATE, MAX_RATE = 0.25, 1.5
        if v < MIN_RATE:
            return MIN_RATE
        if v > MAX_RATE:
            return MAX_RATE
        return v
