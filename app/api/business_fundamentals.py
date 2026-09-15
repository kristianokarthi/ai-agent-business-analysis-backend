from fastapi import APIRouter, HTTPException
from groq import BadRequestError, RateLimitError

from app.agents.business_fundamentals import (
    BusinessFundamentalsAgent,
    InvalidEvidenceReferenceError,
)
from app.schemas.business_fundamentals import (
    BusinessFundamentalsAPIResponse,
    BusinessFundamentalsInput,
)


router = APIRouter(
    prefix="/api/agents/business-fundamentals",
    tags=["Agents"],
)


@router.post("", response_model=BusinessFundamentalsAPIResponse)
async def run_business_fundamentals(
    request: BusinessFundamentalsInput,
) -> BusinessFundamentalsAPIResponse:
    try:
        response = await BusinessFundamentalsAgent().run(request)

        return BusinessFundamentalsAPIResponse(
            result=response.data,
            usage=response.usage,
        )

    except InvalidEvidenceReferenceError as error:
        raise HTTPException(
            status_code=502,
            detail={
                "error_code": "invalid_agent_evidence",
                "message": (
                    "Agent 2 returned an unsupported evidence reference. "
                    "Please retry."
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
