from fastapi import APIRouter, Depends, HTTPException
from groq import (
    APIConnectionError,
    BadRequestError,
    InternalServerError,
    RateLimitError,
)
from pydantic import ValidationError

from app.agents.rag_answer import (
    GroundedAnswerAgent,
    InvalidGroundedAnswerError,
)
from app.embeddings.gemini_provider import (
    GeminiEmbeddingError,
    GeminiEmbeddingConfigurationError,
    GeminiEmbeddingProvider,
    GeminiEmbeddingRateLimitError,
    GeminiEmbeddingRequestError,
)
from app.rag.report_chunker import chunk_strategic_report
from app.rag.semantic_search import search_report_chunks
from app.llm.groq_provider import GroqProvider
from app.schemas.rag import (
    ChunkEmbeddingPreview,
    EmbeddingPreviewRequest,
    EmbeddingPreviewResponse,
    EmbeddingUsage,
    GroundedAnswerRequest,
    GroundedAnswerResponse,
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


def get_groq_provider() -> GroqProvider:
    try:
        return GroqProvider()
    except ValueError as error:
        raise HTTPException(
            status_code=503,
            detail={
                "error_code": "groq_configuration_error",
                "message": str(error),
            },
        ) from error


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


@router.post(
    "/answer/preview",
    response_model=GroundedAnswerResponse,
)
async def preview_grounded_answer(
    request: GroundedAnswerRequest,
    embedding_provider: GeminiEmbeddingProvider = Depends(
        get_embedding_provider
    ),
    llm_provider: GroqProvider = Depends(get_groq_provider),
) -> GroundedAnswerResponse:
    """Retrieve relevant chunks and answer strictly from their evidence."""
    try:
        search_result = await search_report_chunks(
            request,
            embedding_provider,
        )
        generation = await GroundedAnswerAgent(llm_provider).run(
            request.question,
            search_result.matches,
        )
    except GeminiEmbeddingError as error:
        raise embedding_error_to_http(error) from error
    except InvalidGroundedAnswerError as error:
        raise HTTPException(
            status_code=502,
            detail={
                "error_code": "invalid_grounded_answer",
                "message": (
                    "The AI answer referenced unsupported report evidence. "
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
                    "The Groq free-tier limit has been reached. "
                    "Please wait and retry."
                ),
            },
        ) from error
    except (BadRequestError, ValidationError, ValueError) as error:
        raise HTTPException(
            status_code=502,
            detail={
                "error_code": "invalid_grounded_answer_output",
                "message": (
                    "The AI model could not produce a valid grounded answer. "
                    "Please retry."
                ),
            },
        ) from error
    except (APIConnectionError, InternalServerError) as error:
        raise HTTPException(
            status_code=503,
            detail={
                "error_code": "groq_unavailable",
                "message": (
                    "Groq is temporarily unavailable. Please retry later."
                ),
            },
        ) from error

    match_by_id = {
        match.chunk.chunk_id: match
        for match in search_result.matches
    }
    supporting_chunks = [
        match_by_id[chunk_id]
        for chunk_id in generation.data.supporting_chunk_ids
    ]

    evidence_ids: list[str] = []
    sources_by_id = {}
    for match in supporting_chunks:
        for evidence_id in match.chunk.evidence_ids:
            if evidence_id not in evidence_ids:
                evidence_ids.append(evidence_id)
        for source in match.chunk.sources:
            sources_by_id.setdefault(source.evidence_id, source)

    return GroundedAnswerResponse(
        status=generation.data.status,
        question=request.question,
        answer=generation.data.answer,
        supporting_chunks=supporting_chunks,
        evidence_ids=evidence_ids,
        sources=list(sources_by_id.values()),
        limitations=generation.data.limitations,
        retrieval_usage=search_result.usage,
        generation_usage=generation.usage,
    )
