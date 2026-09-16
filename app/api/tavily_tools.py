import secrets

from fastapi import APIRouter, Header, HTTPException

from app.core.config import settings
from app.schemas.web_research import (
    TavilyExtractRequest,
    TavilyExtractResponse,
    TavilySearchRequest,
    TavilySearchResponse,
)
from app.search.tavily_provider import (
    TavilyConfigurationError,
    TavilyProvider,
    TavilyQuotaExceededError,
    TavilyRateLimitError,
    TavilyRequestError,
    TavilyServiceError,
)


router = APIRouter(
    prefix="/api/tools/tavily",
    tags=["Research Tools"],
)


def verify_test_key(
    x_research_tool_key: str | None = Header(default=None),
) -> None:
    expected_key = settings.research_tool_test_key

    if not expected_key:
        raise HTTPException(
            status_code=503,
            detail={
                "error_code": "research_tool_disabled",
                "message": (
                    "Set RESEARCH_TOOL_TEST_KEY to enable the protected "
                    "Swagger research-tool endpoints."
                ),
            },
        )

    if not x_research_tool_key or not secrets.compare_digest(
        x_research_tool_key,
        expected_key,
    ):
        raise HTTPException(
            status_code=403,
            detail={
                "error_code": "invalid_research_tool_key",
                "message": "The research-tool test key is invalid.",
            },
        )


def provider_error_to_http(error: Exception) -> HTTPException:
    if isinstance(error, TavilyConfigurationError):
        return HTTPException(
            status_code=503,
            detail={
                "error_code": "tavily_configuration_error",
                "message": str(error),
            },
        )
    if isinstance(error, TavilyRateLimitError):
        return HTTPException(
            status_code=429,
            detail={
                "error_code": "tavily_rate_limit_exceeded",
                "message": "Tavily is receiving too many requests. Retry later.",
            },
        )
    if isinstance(error, TavilyQuotaExceededError):
        return HTTPException(
            status_code=429,
            detail={
                "error_code": "tavily_credit_limit_reached",
                "message": (
                    "The configured Tavily free-tier credit limit has been "
                    "reached."
                ),
            },
        )
    if isinstance(error, TavilyRequestError):
        return HTTPException(
            status_code=400,
            detail={
                "error_code": "invalid_tavily_request",
                "message": str(error),
            },
        )
    return HTTPException(
        status_code=502,
        detail={
            "error_code": "tavily_unavailable",
            "message": "The web research provider is temporarily unavailable.",
        },
    )


@router.post(
    "/search",
    response_model=TavilySearchResponse,
)
async def test_tavily_search(
    request: TavilySearchRequest,
    x_research_tool_key: str | None = Header(default=None),
) -> TavilySearchResponse:
    verify_test_key(x_research_tool_key)

    try:
        return await TavilyProvider().search(request)
    except (
        TavilyConfigurationError,
        TavilyQuotaExceededError,
        TavilyRateLimitError,
        TavilyRequestError,
        TavilyServiceError,
    ) as error:
        raise provider_error_to_http(error) from error


@router.post(
    "/extract",
    response_model=TavilyExtractResponse,
)
async def test_tavily_extract(
    request: TavilyExtractRequest,
    x_research_tool_key: str | None = Header(default=None),
) -> TavilyExtractResponse:
    verify_test_key(x_research_tool_key)

    try:
        return await TavilyProvider().extract(request)
    except (
        TavilyConfigurationError,
        TavilyQuotaExceededError,
        TavilyRateLimitError,
        TavilyRequestError,
        TavilyServiceError,
    ) as error:
        raise provider_error_to_http(error) from error
