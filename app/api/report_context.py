from fastapi import APIRouter

from app.reports.context_builder import build_report_context
from app.schemas.report_context import ReportContext, ReportContextRequest


router = APIRouter(
    prefix="/api/reports/context",
    tags=["Reports"],
)


@router.post("/preview", response_model=ReportContext)
async def preview_report_context(
    request: ReportContextRequest,
) -> ReportContext:
    """Build the exact compressed, provider-neutral input for Agent 5."""
    return build_report_context(request)

