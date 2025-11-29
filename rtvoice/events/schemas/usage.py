from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, Field


class UsageType(StrEnum):
    TOKENS = "tokens"
    DURATION = "duration"


class TokenInputTokenDetails(BaseModel):
    cached_tokens: int | None = None


class TokenUsage(BaseModel):
    type: Literal[UsageType.TOKENS]
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    input_token_details: TokenInputTokenDetails | None = None


class DurationUsage(BaseModel):
    type: Literal[UsageType.DURATION]
    seconds: float


Usage = Annotated[TokenUsage | DurationUsage, Field(discriminator="type")]
