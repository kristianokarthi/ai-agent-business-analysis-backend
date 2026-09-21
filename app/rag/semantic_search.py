from app.embeddings.gemini_provider import GeminiEmbeddingProvider
from app.schemas.rag import (
    EmbeddingUsage,
    SemanticSearchMatch,
    SemanticSearchRequest,
    SemanticSearchResponse,
)


def cosine_similarity(
    first: list[float],
    second: list[float],
) -> float:
    if len(first) != len(second):
        raise ValueError("Embedding dimensions must match.")

    # Gemini vectors are normalized by the provider, so cosine similarity is
    # their dot product. Clamp tiny floating-point drift to the valid range.
    score = sum(left * right for left, right in zip(first, second, strict=True))
    return max(-1.0, min(1.0, score))


async def search_report_chunks(
    request: SemanticSearchRequest,
    provider: GeminiEmbeddingProvider,
    query_text: str | None = None,
) -> SemanticSearchResponse:
    document_batch = await provider.embed_documents(request.chunks)
    question_batch = await provider.embed_question(
        query_text or request.question
    )
    question_vector = question_batch.vectors[0]

    ranked = sorted(
        (
            (
                cosine_similarity(question_vector, document_vector),
                chunk,
            )
            for chunk, document_vector in zip(
                request.chunks,
                document_batch.vectors,
                strict=True,
            )
        ),
        key=lambda item: item[0],
        reverse=True,
    )[: request.top_k]

    matches = [
        SemanticSearchMatch(
            rank=rank,
            similarity_score=round(score, 6),
            chunk=chunk,
        )
        for rank, (score, chunk) in enumerate(ranked, start=1)
    ]
    return SemanticSearchResponse(
        question=request.question,
        total_chunks_searched=len(request.chunks),
        matches_returned=len(matches),
        matches=matches,
        usage=EmbeddingUsage(
            provider="gemini",
            model=provider.model,
            input_tokens=(
                document_batch.input_tokens
                + question_batch.input_tokens
            ),
            requests=2,
        ),
    )
