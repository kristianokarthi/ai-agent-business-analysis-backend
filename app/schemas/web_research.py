from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, HttpUrl


class StrictSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SearchTopic(str, Enum):
    GENERAL = "general"
    NEWS = "news"
    FINANCE = "finance"


class SearchDepth(str, Enum):
    BASIC = "basic"
    ADVANCED = "advanced"


class TavilySearchRequest(StrictSchema):
    query: str = Field(min_length=3, max_length=300)
    topic: SearchTopic = SearchTopic.GENERAL
    search_depth: SearchDepth = SearchDepth.BASIC
    max_results: int = Field(default=5, ge=1, le=5)
    country: str | None = Field(default=None, min_length=2, max_length=50)
    time_range: str | None = Field(
        default=None,
        pattern="^(day|week|month|year)$",
    )


class SearchResult(StrictSchema):
    title: str
    url: HttpUrl
    content: str
    relevance_score: float = Field(ge=0)
    published_date: str | None = None


class TavilyUsage(StrictSchema):
    provider: str = "tavily"
    operation: str
    credits_used: float = Field(ge=0)
    response_time_seconds: float = Field(ge=0)


class TavilySearchResponse(StrictSchema):
    query: str
    results: list[SearchResult]
    usage: TavilyUsage
    request_id: str | None = None


class TavilyExtractRequest(StrictSchema):
    urls: list[HttpUrl] = Field(min_length=1, max_length=5)
    query: str = Field(min_length=3, max_length=300)
    extract_depth: SearchDepth = SearchDepth.BASIC
    chunks_per_source: int = Field(default=3, ge=1, le=5)


class ExtractedDocument(StrictSchema):
    url: HttpUrl
    content: str


class FailedExtraction(StrictSchema):
    url: HttpUrl
    error: str


class TavilyExtractResponse(StrictSchema):
    documents: list[ExtractedDocument]
    failed_documents: list[FailedExtraction]
    usage: TavilyUsage
    request_id: str | None = None
