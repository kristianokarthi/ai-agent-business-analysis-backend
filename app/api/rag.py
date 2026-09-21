from fastapi import APIRouter, Depends, HTTPException

from app.embeddings.gemini_provider import (
    GeminiEmbeddingConfigurationError,
    GeminiEmbeddingProvider,
    GeminiEmbeddingRateLimitError,
    GeminiEmbeddingRequestError,
    GeminiEmbeddingServiceError,
)
from app.rag.report_chunker import chunk_strategic_report
from app.schemas.rag import (
    ChunkEmbeddingPreview,
    EmbeddingPreviewRequest,
    EmbeddingPreviewResponse,
    EmbeddingUsage,
    ReportChunkPreviewRequest,
    ReportChunkPreviewResponse,
)


router = APIRouter(
    prefix="/api/rag",
    tags=["RAG Learning"],
)


def get_embedding_provider() -> GeminiEmbeddingProvider:
    return GeminiEmbeddingProvider()


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
    except GeminiEmbeddingConfigurationError as error:
        raise HTTPException(
            status_code=503,
            detail={
                "error_code": "gemini_embedding_configuration_error",
                "message": str(error),
            },
        ) from error
    except GeminiEmbeddingRateLimitError as error:
        raise HTTPException(
            status_code=429,
            detail={
                "error_code": "gemini_embedding_rate_limit_exceeded",
                "message": str(error),
            },
        ) from error
    except GeminiEmbeddingRequestError as error:
        raise HTTPException(
            status_code=400,
            detail={
                "error_code": "invalid_gemini_embedding_request",
                "message": str(error),
            },
        ) from error
    except GeminiEmbeddingServiceError as error:
        raise HTTPException(
            status_code=503,
            detail={
                "error_code": "gemini_embedding_unavailable",
                "message": str(error),
            },
        ) from error

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
