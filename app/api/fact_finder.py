from fastapi import APIRouter, HTTPException
from groq import BadRequestError, RateLimitError

from app.agents.fact_finder import FactFinderAgent
from app.schemas.fact_finder import (
    FactFinderAPIResponse,
    FactFinderInput,
)

router = APIRouter(
    prefix="/api/agents/fact-finder",
    tags=["Agents"],
)


@router.post("", response_model=FactFinderAPIResponse)
async def run_fact_finder(
    request: FactFinderInput,
) -> FactFinderAPIResponse:
    try:
        agent = FactFinderAgent()
        response = await agent.run(request)

        return FactFinderAPIResponse(
            result=response.data,
            usage=response.usage,
        )

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
            groq_error = error.body.get("error", {})
            error_code = groq_error.get("code")

        if error_code == "json_validate_failed":
            raise HTTPException(
                status_code=502,
                detail={
                    "error_code": "invalid_llm_json",
                    "message": (
                        "The AI model could not generate a valid "
                        "structured response. Please retry the request."
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