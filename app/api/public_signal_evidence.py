from fastapi import APIRouter, Header

from app.api.tavily_tools import (
    provider_error_to_http,
    verify_test_key,
)
from app.collectors.public_signal_collector import PublicSignalCollector
from app.schemas.public_signal_evidence import (
    PublicSignalCollectionInput,
    PublicSignalCollectionOutput,
)
from app.search.tavily_provider import (
    TavilyConfigurationError,
    TavilyQuotaExceededError,
    TavilyRateLimitError,
    TavilyRequestError,
    TavilyServiceError,
)


router = APIRouter(
    prefix="/api/collectors/public-signals",
    tags=["Research Collectors"],
)


@router.post("", response_model=PublicSignalCollectionOutput)
async def collect_public_signals(
    request: PublicSignalCollectionInput,
    x_research_tool_key: str | None = Header(default=None),
) -> PublicSignalCollectionOutput:
    """Protected Swagger endpoint for testing collection without Groq."""
    verify_test_key(x_research_tool_key)

    try:
        return await PublicSignalCollector().collect(request)
    except (
        TavilyConfigurationError,
        TavilyQuotaExceededError,
        TavilyRateLimitError,
        TavilyRequestError,
        TavilyServiceError,
    ) as error:
        raise provider_error_to_http(error) from error
