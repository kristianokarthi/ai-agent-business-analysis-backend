from fastapi import APIRouter, Depends, HTTPException

from app.embeddings.gemini_provider import (
    GeminiEmbeddingError,
    GeminiEmbeddingConfigurationError,
    GeminiEmbeddingProvider,
    GeminiEmbeddingRateLimitError,
    GeminiEmbeddingRequestError,
)
from app.rag.report_chunker import chunk_strategic_report
from app.rag.semantic_search import search_report_chunks
from app.schemas.rag import (
    ChunkEmbeddingPreview,
    EmbeddingPreviewRequest,
    EmbeddingPreviewResponse,
    EmbeddingUsage,
    ReportChunkPreviewRequest,
    ReportChunkPreviewResponse,
    SemanticSearchRequest,
    SemanticSearchResponse,
)


router = APIRouter(
    prefix="/api/rag",
    tags=["RAG Learning"],
)


def get_embedding_provider() -> GeminiEmbeddingProvider:
    return GeminiEmbeddingProvider()


def embedding_error_to_http(error: GeminiEmbeddingError) -> HTTPException:
    if isinstance(error, GeminiEmbeddingConfigurationError):
        return HTTPException(
            status_code=503,
            detail={
                "error_code": "gemini_embedding_configuration_error",
                "message": str(error),
            },
        )
    if isinstance(error, GeminiEmbeddingRateLimitError):
        return HTTPException(
            status_code=429,
            detail={
                "error_code": "gemini_embedding_rate_limit_exceeded",
                "message": str(error),
            },
        )
    if isinstance(error, GeminiEmbeddingRequestError):
        return HTTPException(
            status_code=400,
            detail={
                "error_code": "invalid_gemini_embedding_request",
                "message": str(error),
            },
        )
    return HTTPException(
        status_code=503,
        detail={
            "error_code": "gemini_embedding_unavailable",
            "message": str(error),
        },
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


@router.post(
    "/embeddings/preview",
    response_model=EmbeddingPreviewResponse,
)
async def preview_chunk_embeddings(
    request: EmbeddingPreviewRequest,
    provider: GeminiEmbeddingProvider = Depends(get_embedding_provider),
) -> EmbeddingPreviewResponse:
    """Generate document embeddings while exposing only a readable preview."""
    try:
        batch = await provider.embed_documents(request.chunks)
    except GeminiEmbeddingError as error:
        raise embedding_error_to_http(error) from error

    embeddings = [
        ChunkEmbeddingPreview(
            chunk_id=chunk.chunk_id,
            section=chunk.section,
            title=chunk.title,
            dimensions=len(vector),
            vector_preview=vector[:8],
        )
        for chunk, vector in zip(request.chunks, batch.vectors, strict=True)
    ]
    return EmbeddingPreviewResponse(
        total_embeddings=len(embeddings),
        dimensions=provider.dimensions,
        embeddings=embeddings,
        usage=EmbeddingUsage(
            provider="gemini",
            model=provider.model,
            input_tokens=batch.input_tokens,
            requests=1,
        ),
    )


@router.post(
    "/search/preview",
    response_model=SemanticSearchResponse,
)
async def preview_semantic_search(
    request: SemanticSearchRequest,
    provider: GeminiEmbeddingProvider = Depends(get_embedding_provider),
) -> SemanticSearchResponse:
    """Rank supplied report chunks by semantic relevance to a question."""
    try:
        return await search_report_chunks(request, provider)
    except GeminiEmbeddingError as error:
        raise embedding_error_to_http(error) from error
