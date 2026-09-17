from fastapi import APIRouter, HTTPException
from groq import BadRequestError, RateLimitError
from pydantic import ValidationError

from app.api.tavily_tools import provider_error_to_http
from app.agents.customer_reputation import (
    CustomerReputationAgent,
    InvalidPublicSignalReferenceError,
)
from app.collectors.public_signal_collector import PublicSignalCollector
from app.schemas.customer_reputation import (
    CustomerReputationAPIResponse,
    CustomerReputationInput,
)
from app.schemas.public_signal_evidence import (
    CustomerReputationResearchResponse,
    PublicSignalCollectionInput,
    PublicSignalCollectionStatus,
)
from app.search.tavily_provider import (
    TavilyConfigurationError,
    TavilyQuotaExceededError,
    TavilyRateLimitError,
    TavilyRequestError,
    TavilyServiceError,
)


router = APIRouter(
    prefix="/api/agents/customer-reputation",
    tags=["Agents"],
)


@router.post(
    "/research",
    response_model=CustomerReputationResearchResponse,
)
async def research_customer_reputation(
    request: PublicSignalCollectionInput,
) -> CustomerReputationResearchResponse:
    """Collect current public signals and run Agent 4 with them."""
    try:
        collection = await PublicSignalCollector().collect(request)
    except (
        TavilyConfigurationError,
        TavilyQuotaExceededError,
        TavilyRateLimitError,
        TavilyRequestError,
        TavilyServiceError,
    ) as error:
        raise provider_error_to_http(error) from error

    if (
        collection.status
        == PublicSignalCollectionStatus.INSUFFICIENT_SOURCES
    ):
        raise HTTPException(
            status_code=422,
            detail={
                "error_code": "insufficient_public_signals",
                "message": (
                    "No usable public signals were collected, so Agent 4 "
                    "was not run."
                ),
                "warnings": collection.warnings,
            },
        )

    agent_response = await run_customer_reputation(
        CustomerReputationInput(
            company_name=request.company_name,
            purpose=request.purpose,
            follow_up_answers=request.follow_up_answers,
            signal_documents=collection.signal_documents,
        )
    )
    return CustomerReputationResearchResponse(
        result=agent_response.result,
        usage=agent_response.usage,
        collection=collection,
    )


@router.post("", response_model=CustomerReputationAPIResponse)
async def run_customer_reputation(
    request: CustomerReputationInput,
) -> CustomerReputationAPIResponse:
    try:
        response = await CustomerReputationAgent().run(request)

        return CustomerReputationAPIResponse(
            result=response.data,
            usage=response.usage,
        )

    except InvalidPublicSignalReferenceError as error:
        raise HTTPException(
            status_code=502,
            detail={
                "error_code": "invalid_public_signal_reference",
                "message": (
                    "Agent 4 omitted or cited an unsupported public "
                    "signal. Please retry."
                ),
            },
        ) from error

    except ValidationError as error:
        raise HTTPException(
            status_code=502,
            detail={
                "error_code": "invalid_llm_output",
                "message": (
                    "Agent 4 produced inconsistent sentiment counts or "
                    "theme data. Please retry."
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
