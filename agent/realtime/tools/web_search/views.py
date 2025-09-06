from typing import Optional
from pydantic import BaseModel


class TavilyResult(BaseModel):
    url: str
    title: str
    content: Optional[str] = None
    score: float
    raw_content: Optional[str] = None


class TavilyResponse(BaseModel):
    query: str
    follow_up_questions: Optional[list[str]] = None
    answer: Optional[str] = None
    images: list[str] = []
    results: list[TavilyResult]
    response_time: float
    request_id: str
