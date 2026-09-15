from fastapi import APIRouter, HTTPException
from groq import BadRequestError, RateLimitError

from app.agents.market_competitor import (
    InvalidMarketEvidenceReferenceError,
    MarketCompetitorAgent,
)
from app.schemas.market_competitor import (
    MarketCompetitorAPIResponse,
    MarketCompetitorInput,
)


router = APIRouter(
    prefix="/api/agents/market-competitor",
    tags=["Agents"],
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
