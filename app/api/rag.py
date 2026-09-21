from fastapi import APIRouter, Depends, HTTPException

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
from app.core.config import settings
from app.llm.openrouter_provider import (
    OpenRouterConfigurationError,
    OpenRouterCreditError,
    OpenRouterInvalidResponseError,
    OpenRouterProvider,
    OpenRouterRateLimitError,
    OpenRouterRequestError,
    OpenRouterServiceError,
    OpenRouterTruncatedResponseError,
)
from app.rag.conversation import build_contextual_query
from app.rag.report_chunker import chunk_strategic_report
from app.rag.semantic_search import search_report_chunks
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


def get_chat_provider() -> OpenRouterProvider:
    try:
        return OpenRouterProvider(model=settings.openrouter_chat_model)
    except OpenRouterConfigurationError as error:
        raise HTTPException(
            status_code=503,
            detail={
                "error_code": "openrouter_configuration_error",
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
            requests=batch.requests,
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
    llm_provider: OpenRouterProvider = Depends(get_chat_provider),
) -> GroundedAnswerResponse:
    """Retrieve relevant chunks and answer strictly from their evidence."""
    try:
        contextual_query = build_contextual_query(
            request.question,
            request.conversation_history,
        )
        search_result = await search_report_chunks(
            request,
            embedding_provider,
            query_text=contextual_query,
        )
        if (
            search_result.matches[0].similarity_score
            < settings.rag_min_similarity
        ):
            return GroundedAnswerResponse(
                status="insufficient_evidence",
                question=request.question,
                answer=(
                    "I couldn't find enough relevant information in this report "
                    "to answer that. Try naming the finding or risk you mean, "
                    "or ask about the report's customer feedback."
                ),
                supporting_chunks=[],
                evidence_ids=[],
                sources=[],
                limitations=[
                    "The available report sections did not provide a sufficiently relevant match."
                ],
                retrieval_usage=search_result.usage,
                generation_usage=None,
            )
        generation = await GroundedAnswerAgent(llm_provider).run(
            request.question,
            search_result.matches,
            request.conversation_history,
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
    except OpenRouterRateLimitError as error:
        raise HTTPException(
            status_code=429,
            detail={
                "error_code": "openrouter_rate_limit_exceeded",
                "message": (
                    "The OpenRouter free-tier limit has been reached. "
                    "Please wait and retry."
                ),
            },
        ) from error
    except (
        OpenRouterInvalidResponseError,
        OpenRouterTruncatedResponseError,
    ) as error:
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
    except OpenRouterCreditError as error:
        raise HTTPException(
            status_code=402,
            detail={
                "error_code": "openrouter_credit_error",
                "message": str(error),
            },
        ) from error
    except OpenRouterRequestError as error:
        raise HTTPException(
            status_code=400,
            detail={
                "error_code": "openrouter_bad_request",
                "message": str(error),
            },
        ) from error
    except OpenRouterServiceError as error:
        raise HTTPException(
            status_code=503,
            detail={
                "error_code": "openrouter_unavailable",
                "message": (
                    "OpenRouter is temporarily unavailable. Please retry later."
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
