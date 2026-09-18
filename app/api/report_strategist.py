import logging

from fastapi import APIRouter, HTTPException
from pydantic import ValidationError

from app.agents.report_strategist import (
    InvalidReportEvidenceError,
    ReportStrategistAgent,
    UnsafeStockRecommendationError,
    context_metrics,
)
from app.llm.openrouter_provider import (
    OpenRouterConfigurationError,
    OpenRouterCreditError,
    OpenRouterInvalidResponseError,
    OpenRouterRateLimitError,
    OpenRouterRequestError,
    OpenRouterServiceError,
    OpenRouterTruncatedResponseError,
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

logger = logging.getLogger("uvicorn.error")


async def _run(context) -> ReportStrategistAPIResponse:
    try:
        response = await ReportStrategistAgent().run(context)
        return ReportStrategistAPIResponse(
            result=response.data,
            usage=response.usage,
            context=context_metrics(context),
        )
    except UnsafeStockRecommendationError as error:
        logger.warning(
            "Agent 5 stock-safety validation failed | reason=%s",
            error,
        )
        raise HTTPException(
            status_code=502,
            detail={
                "error_code": "unsafe_stock_recommendation",
                "message": (
                    "Agent 5 produced a direct stock-trading instruction. "
                    "Please retry."
                ),
            },
        ) from error
    except InvalidReportEvidenceError as error:
        logger.warning(
            "Agent 5 evidence validation failed | reason=%s",
            error,
        )
        raise HTTPException(
            status_code=502,
            detail={
                "error_code": "invalid_report_evidence",
                "message": str(error),
            },
        ) from error
    except OpenRouterTruncatedResponseError as error:
        raise HTTPException(
            status_code=502,
            detail={
                "error_code": "openrouter_output_truncated",
                "message": (
                    "OpenRouter reached the Agent 5 output-token limit before "
                    "finishing the report."
                ),
            },
        ) from error
    except (OpenRouterInvalidResponseError, ValidationError) as error:
        raise HTTPException(
            status_code=502,
            detail={
                "error_code": "invalid_openrouter_output",
                "message": (
                    "OpenRouter could not produce a valid structured report. "
                    "Please retry."
                ),
            },
        ) from error
    except OpenRouterRateLimitError as error:
        raise HTTPException(
            status_code=429,
            detail={
                "error_code": "openrouter_rate_limit_exceeded",
                "message": (
                    "The OpenRouter request limit has been reached. "
                    "Please wait and try again later."
                ),
            },
        ) from error
    except (OpenRouterConfigurationError, OpenRouterCreditError) as error:
        raise HTTPException(
            status_code=503,
            detail={
                "error_code": "openrouter_configuration_error",
                "message": (
                    "OpenRouter is not configured correctly or does not have "
                    "enough credit."
                ),
            },
        ) from error
    except OpenRouterRequestError as error:
        raise HTTPException(
            status_code=400,
            detail={
                "error_code": "openrouter_bad_request",
                "message": "OpenRouter rejected the report request.",
            },
        ) from error
    except OpenRouterServiceError as error:
        raise HTTPException(
            status_code=503,
            detail={
                "error_code": "openrouter_unavailable",
                "message": "OpenRouter is temporarily unavailable.",
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
