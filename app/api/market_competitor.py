from fastapi import APIRouter, HTTPException
from groq import BadRequestError, RateLimitError

from app.api.tavily_tools import provider_error_to_http
from app.agents.market_competitor import (
    InvalidMarketEvidenceReferenceError,
    MarketCompetitorAgent,
)
from app.collectors.market_collector import MarketEvidenceCollector
from app.schemas.market_competitor import (
    MarketCompetitorAPIResponse,
    MarketCompetitorInput,
)
from app.schemas.market_evidence import (
    MarketCollectionStatus,
    MarketCompetitorResearchResponse,
    MarketEvidenceCollectionInput,
)
from app.search.tavily_provider import (
    TavilyConfigurationError,
    TavilyQuotaExceededError,
    TavilyRateLimitError,
    TavilyRequestError,
    TavilyServiceError,
)


router = APIRouter(
    prefix="/api/agents/market-competitor",
    tags=["Agents"],
)


@router.post(
    "/research",
    response_model=MarketCompetitorResearchResponse,
)
async def research_market_competitors(
    request: MarketEvidenceCollectionInput,
) -> MarketCompetitorResearchResponse:
    """Collect current market sources and run Agent 3 with them."""
    try:
        collection = await MarketEvidenceCollector().collect(request)
    except (
        TavilyConfigurationError,
        TavilyQuotaExceededError,
        TavilyRateLimitError,
        TavilyRequestError,
        TavilyServiceError,
    ) as error:
        raise provider_error_to_http(error) from error

    if collection.status == MarketCollectionStatus.INSUFFICIENT_SOURCES:
        raise HTTPException(
            status_code=422,
            detail={
                "error_code": "insufficient_market_sources",
                "message": (
                    "No usable market sources were collected, so Agent 3 "
                    "was not run."
                ),
                "warnings": collection.warnings,
            },
        )

    agent_response = await run_market_competitor(
        MarketCompetitorInput(
            purpose=request.purpose,
            follow_up_answers=request.follow_up_answers,
            company_evidence=request.company_evidence,
            business_fundamentals=request.business_fundamentals,
            market_documents=collection.market_documents,
        )
    )

    return MarketCompetitorResearchResponse(
        result=agent_response.result,
        usage=agent_response.usage,
        collection=collection,
    )


@router.post("", response_model=MarketCompetitorAPIResponse)
async def run_market_competitor(
    request: MarketCompetitorInput,
) -> MarketCompetitorAPIResponse:
    try:
        response = await MarketCompetitorAgent().run(request)

        return MarketCompetitorAPIResponse(
            result=response.data,
            usage=response.usage,
        )

    except InvalidMarketEvidenceReferenceError as error:
        raise HTTPException(
            status_code=502,
            detail={
                "error_code": "invalid_market_evidence",
                "message": (
                    "Agent 3 returned an unsupported evidence reference "
                    "or company identity. Please retry."
                ),
            },
        ) from error

    except RateLimitError as error:
        raise HTTPException(
            status_code=429,
            detail={
                "error_code": "groq_rate_limit_exceeded",
                "message": (
                    "The Groq free-tier usage limit has been reached. "
                    "Please wait and try again later."
                ),
            },
        ) from error

    except BadRequestError as error:
        error_code = None
        if isinstance(error.body, dict):
            error_code = error.body.get("error", {}).get("code")

        if error_code == "json_validate_failed":
            raise HTTPException(
                status_code=502,
                detail={
                    "error_code": "invalid_llm_json",
                    "message": (
                        "The AI model could not generate a valid "
                        "structured response. Please retry."
                    ),
                },
            ) from error

        raise HTTPException(
            status_code=400,
            detail={
                "error_code": "groq_bad_request",
                "message": "Groq rejected the AI request.",
            },
        ) from error

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail={
                "error_code": "internal_server_error",
                "message": "An unexpected server error occurred.",
            },
        ) from error
