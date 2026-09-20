from fastapi import APIRouter

from app.rag.report_chunker import chunk_strategic_report
from app.schemas.rag import (
    ReportChunkPreviewRequest,
    ReportChunkPreviewResponse,
)


router = APIRouter(
    prefix="/api/rag",
    tags=["RAG Learning"],
)


@router.post(
    "/chunks/preview",
    response_model=ReportChunkPreviewResponse,
)
async def preview_report_chunks(
    request: ReportChunkPreviewRequest,
) -> ReportChunkPreviewResponse:
    """Preview deterministic chunks before adding embeddings or storage."""
    return chunk_strategic_report(request.report)
