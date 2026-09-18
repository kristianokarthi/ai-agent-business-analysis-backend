from fastapi import APIRouter, HTTPException
from pydantic import ValidationError

from app.agents.report_strategist import (
    InvalidReportEvidenceError,
    ReportStrategistAgent,
    UnsafeStockRecommendationError,
    context_metrics,
)
from app.llm.gemini_provider import (
    GeminiConfigurationError,
    GeminiInvalidResponseError,
    GeminiRateLimitError,
    GeminiRequestError,
    GeminiServiceError,
)
from app.reports.context_builder import build_report_context
from app.schemas.report_strategist import (
    ReportStrategistAPIResponse,
    ReportStrategistContextRequest,
    ReportStrategistResearchRequest,
)


router = APIRouter(
    prefix="/api/agents/report-strategist",
    tags=["Agents"],
)


async def _run(context) -> ReportStrategistAPIResponse:
    try:
        response = await ReportStrategistAgent().run(context)
        return ReportStrategistAPIResponse(
            result=response.data,
            usage=response.usage,
            context=context_metrics(context),
        )
    except (InvalidReportEvidenceError, UnsafeStockRecommendationError) as error:
        raise HTTPException(
            status_code=502,
            detail={
                "error_code": "invalid_report_evidence",
                "message": (
                    "Agent 5 produced an unsupported citation, identity, purpose, "
                    "or recommendation. Please retry."
                ),
            },
        ) from error
    except (GeminiInvalidResponseError, ValidationError) as error:
        raise HTTPException(
            status_code=502,
            detail={
                "error_code": "invalid_gemini_output",
                "message": (
                    "Gemini could not produce a valid structured report. "
                    "Please retry."
                ),
            },
        ) from error
    except GeminiRateLimitError as error:
        raise HTTPException(
            status_code=429,
            detail={
                "error_code": "gemini_rate_limit_exceeded",
                "message": (
                    "The Gemini project rate or daily limit has been reached. "
                    "Please wait and try again later."
                ),
            },
        ) from error
    except GeminiConfigurationError as error:
        raise HTTPException(
            status_code=503,
            detail={
                "error_code": "gemini_configuration_error",
                "message": "Gemini is not configured correctly on the backend.",
            },
        ) from error
    except GeminiRequestError as error:
        raise HTTPException(
            status_code=400,
            detail={
                "error_code": "gemini_bad_request",
                "message": "Gemini rejected the report request.",
            },
        ) from error
    except GeminiServiceError as error:
        raise HTTPException(
            status_code=503,
            detail={
                "error_code": "gemini_unavailable",
                "message": "Gemini is temporarily unavailable.",
            },
        ) from error


@router.post("", response_model=ReportStrategistAPIResponse)
async def run_report_strategist(
    request: ReportStrategistContextRequest,
) -> ReportStrategistAPIResponse:
    """Run Agent 5 with an already-compressed ReportContext."""
    return await _run(request.context)


@router.post("/research", response_model=ReportStrategistAPIResponse)
async def research_report_strategist(
    request: ReportStrategistResearchRequest,
) -> ReportStrategistAPIResponse:
    """Build compressed context from Agents 1–4 and run Agent 5."""
    return await _run(build_report_context(request))
