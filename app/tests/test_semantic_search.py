import json

import httpx
import pytest
from fastapi.testclient import TestClient

from app.api.rag import get_embedding_provider
from app.embeddings.gemini_provider import GeminiEmbeddingProvider
from app.main import app
from app.rag.conversation import build_contextual_query
from app.rag.semantic_search import cosine_similarity, search_report_chunks
from app.schemas.rag import (
    ReportChunk,
    ReportChunkSection,
    SemanticSearchRequest,
    ChatMessage,
)


def searchable_chunks() -> list[ReportChunk]:
    return [
        ReportChunk(
            chunk_id="risk_01",
            section=ReportChunkSection.RISK,
            title="Raw Material Cost Volatility",
            content="Higher material prices may reduce manufacturing margins.",
        ),
        ReportChunk(
            chunk_id="risk_02",
            section=ReportChunkSection.RISK,
            title="Post-Sales Service Concerns",
            content="Customers report inadequate post-sales support.",
            evidence_ids=["signal_review_1"],
        ),
    ]


def semantic_transport() -> httpx.MockTransport:
    async def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        task_type = payload["requests"][0]["taskType"]
        if task_type == "RETRIEVAL_DOCUMENT":
            return httpx.Response(
                200,
                json={
                    "embeddings": [
                        {"values": [1, 0, 0]},
                        {"values": [0, 1, 0]},
                    ],
                    "usageMetadata": {"promptTokenCount": 20},
                },
            )
        return httpx.Response(
            200,
            json={
                "embeddings": [{"values": [0, 2, 0]}],
                "usageMetadata": {"promptTokenCount": 5},
            },
        )

    return httpx.MockTransport(handler)


def test_cosine_similarity_uses_normalized_vectors():
    assert cosine_similarity([1, 0], [1, 0]) == 1
    assert cosine_similarity([1, 0], [0, 1]) == 0


def test_follow_up_query_includes_recent_conversation():
    query = build_contextual_query(
        "How could that affect the company?",
        [
            ChatMessage(
                role="user",
                content="What customer-service issue was reported?",
            ),
            ChatMessage(
                role="assistant",
                content="The report mentions inadequate post-sales support.",
            ),
        ],
    )

    assert "inadequate post-sales support" in query
    assert "How could that affect the company?" in query


@pytest.mark.anyio
async def test_semantic_search_ranks_relevant_chunk_first():
    provider = GeminiEmbeddingProvider(
        api_key="test-key",
        dimensions=3,
        transport=semantic_transport(),
    )
    request = SemanticSearchRequest(
        question="What customer service problems were reported?",
        chunks=searchable_chunks(),
        top_k=1,
    )

    result = await search_report_chunks(request, provider)

    assert result.matches_returned == 1
    assert result.matches[0].chunk.chunk_id == "risk_02"
    assert result.matches[0].similarity_score == 1
    assert result.matches[0].chunk.evidence_ids == ["signal_review_1"]
    assert result.usage.requests == 2
    assert result.usage.input_tokens == 25


def test_semantic_search_preview_endpoint():
    provider = GeminiEmbeddingProvider(
        api_key="test-key",
        dimensions=3,
        transport=semantic_transport(),
    )
    app.dependency_overrides[get_embedding_provider] = lambda: provider
    try:
        response = TestClient(app).post(
            "/api/rag/search/preview",
            json={
                "question": "What customer service problems were reported?",
                "chunks": [
                    chunk.model_dump(mode="json")
                    for chunk in searchable_chunks()
                ],
                "top_k": 2,
            },
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    payload = response.json()
    assert payload["total_chunks_searched"] == 2
    assert payload["matches"][0]["chunk"]["chunk_id"] == "risk_02"
    assert payload["matches"][1]["chunk"]["chunk_id"] == "risk_01"
