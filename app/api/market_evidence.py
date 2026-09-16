from fastapi import APIRouter, Header

from app.api.tavily_tools import (
    provider_error_to_http,
    verify_test_key,
)
from app.collectors.market_collector import MarketEvidenceCollector
from app.schemas.market_evidence import (
    MarketEvidenceCollectionInput,
    MarketEvidenceCollectionOutput,
)
from app.search.tavily_provider import (
    TavilyConfigurationError,
    TavilyQuotaExceededError,
    TavilyRateLimitError,
    TavilyRequestError,
    TavilyServiceError,
)


router = APIRouter(
    prefix="/api/collectors/market-evidence",
    tags=["Research Collectors"],
)


@router.post("", response_model=MarketEvidenceCollectionOutput)
async def collect_market_evidence(
    request: MarketEvidenceCollectionInput,
    x_research_tool_key: str | None = Header(default=None),
) -> MarketEvidenceCollectionOutput:
    """Protected Swagger endpoint for testing collection without Groq."""
    verify_test_key(x_research_tool_key)

    try:
        return await MarketEvidenceCollector().collect(request)
    except (
        TavilyConfigurationError,
        TavilyQuotaExceededError,
        TavilyRateLimitError,
        TavilyRequestError,
        TavilyServiceError,
    ) as error:
        raise provider_error_to_http(error) from error
