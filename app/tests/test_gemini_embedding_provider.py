import json
import math

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.api.rag import get_embedding_provider
from app.embeddings.gemini_provider import (
    GeminiEmbeddingConfigurationError,
    GeminiEmbeddingProvider,
    GeminiEmbeddingRateLimitError,
)
from app.main import app
from app.schemas.rag import (
    EmbeddingPreviewRequest, GroundedAnswerRequest,
    ReportChunk, ReportChunkSection, SemanticSearchRequest,
)
from app.rag.semantic_search import search_report_chunks


@pytest.mark.parametrize("schema", [
    EmbeddingPreviewRequest, SemanticSearchRequest, GroundedAnswerRequest,
])
def test_chunk_limit_accepts_120_and_rejects_121(schema):
    payload = {"chunks": chunks() * 60}
    if schema is not EmbeddingPreviewRequest:
        payload["question"] = "What are the risks?"
    assert len(schema(**payload).chunks) == 120
    payload["chunks"].append(chunks()[0])
    with pytest.raises(ValidationError):
        schema(**payload)


@pytest.mark.anyio
async def test_large_report_batches_preserve_ranking_and_usage():
    sizes = []
    offset = 0

    async def handler(request):
        nonlocal offset
        items = json.loads(request.content)["requests"]
        sizes.append(len(items))
        if items[0]["taskType"] == "QUESTION_ANSWERING":
            values = [[120, 1]]
        else:
            values = [[index + 1, 1] for index in range(offset, offset + len(items))]
            offset += len(items)
        return httpx.Response(200, json={
            "embeddings": [{"values": value} for value in values],
            "usageMetadata": {"promptTokenCount": len(items) * 2},
        })

    provider = GeminiEmbeddingProvider(
        api_key="test-key", dimensions=2, transport=httpx.MockTransport(handler),
    )
    report_chunks = [
        chunks()[0].model_copy(update={"chunk_id": f"risk_{index}"})
        for index in range(120)
    ]
    result = await search_report_chunks(
        SemanticSearchRequest(question="What are the risks?", chunks=report_chunks),
        provider,
    )
    assert sizes == [50, 50, 20, 1]
    assert result.matches[0].chunk.chunk_id == "risk_119"
    assert result.total_chunks_searched == 120
    assert result.matches_returned == 3
    assert result.usage.requests == 4
    assert result.usage.input_tokens == 242


@pytest.mark.anyio
async def test_failed_second_batch_does_not_return_partial_embeddings():
    sizes = []

    async def handler(request):
        items = json.loads(request.content)["requests"]
        sizes.append(len(items))
        if len(sizes) == 2:
            return httpx.Response(429)
        return httpx.Response(200, json={
            "embeddings": [{"values": [1, 0]} for _ in items],
        })

    provider = GeminiEmbeddingProvider(
        api_key="test-key", dimensions=2, transport=httpx.MockTransport(handler),
    )
    with pytest.raises(GeminiEmbeddingRateLimitError):
        await provider.embed_documents(chunks() * 60)
    assert sizes == [50, 50]


def chunks() -> list[ReportChunk]:
    return [
        ReportChunk(
            chunk_id="risk_01",
            section=ReportChunkSection.RISK,
            title="Raw Material Cost Volatility",
            content="Raw-material prices may reduce manufacturing margins.",
        ),
        ReportChunk(
            chunk_id="opportunity_01",
            section=ReportChunkSection.OPPORTUNITY,
            title="Electric Vehicle Growth",
            content="Electric vehicles may provide a growth opportunity.",
        ),
    ]


@pytest.mark.anyio
async def test_provider_generates_normalized_document_embeddings():
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["x-goog-api-key"] == "test-key"
        payload = json.loads(request.content)
        assert len(payload["requests"]) == 2
        assert payload["requests"][0]["taskType"] == "RETRIEVAL_DOCUMENT"
        assert payload["requests"][0]["outputDimensionality"] == 3
        return httpx.Response(
            200,
            json={
                "embeddings": [
                    {"values": [3, 4, 0]},
                    {"values": [0, 0, 2]},
                ],
                "usageMetadata": {"promptTokenCount": 24},
            },
        )

    provider = GeminiEmbeddingProvider(
        api_key="test-key",
        dimensions=3,
        transport=httpx.MockTransport(handler),
    )
    result = await provider.embed_documents(chunks())

    assert result.input_tokens == 24
    assert result.vectors[0] == pytest.approx([0.6, 0.8, 0])
    assert math.sqrt(sum(value**2 for value in result.vectors[1])) == pytest.approx(1)


@pytest.mark.anyio
async def test_provider_maps_rate_limit_error():
    transport = httpx.MockTransport(
        lambda request: httpx.Response(429, json={"error": "limited"})
    )
    provider = GeminiEmbeddingProvider(
        api_key="test-key",
        dimensions=3,
        transport=transport,
    )

    with pytest.raises(GeminiEmbeddingRateLimitError):
        await provider.embed_documents(chunks())


@pytest.mark.anyio
async def test_provider_embeds_question_for_question_answering():
    async def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert len(payload["requests"]) == 1
        assert payload["requests"][0]["taskType"] == "QUESTION_ANSWERING"
        assert payload["requests"][0]["content"]["parts"][0]["text"] == (
            "What could reduce profitability?"
        )
        return httpx.Response(
            200,
            json={
                "embeddings": [{"values": [2, 0, 0]}],
                "usageMetadata": {"promptTokenCount": 6},
            },
        )

    provider = GeminiEmbeddingProvider(
        api_key="test-key",
        dimensions=3,
        transport=httpx.MockTransport(handler),
    )
    result = await provider.embed_question(
        "What could reduce profitability?"
    )

    assert result.vectors[0] == [1, 0, 0]
    assert result.input_tokens == 6


def test_provider_requires_api_key():
    with pytest.raises(GeminiEmbeddingConfigurationError):
        GeminiEmbeddingProvider(api_key="")


def test_embedding_preview_endpoint():
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "embeddings": [
                    {"values": [3, 4, 0]},
                    {"values": [0, 0, 2]},
                ],
                "usageMetadata": {"promptTokenCount": 24},
            },
        )

    provider = GeminiEmbeddingProvider(
        api_key="test-key",
        dimensions=3,
        transport=httpx.MockTransport(handler),
    )
    app.dependency_overrides[get_embedding_provider] = lambda: provider
    try:
        client = TestClient(app)
        response = client.post(
            "/api/rag/embeddings/preview",
            json={
                "chunks": [
                    chunk.model_dump(mode="json")
                    for chunk in chunks()
                ]
            },
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    payload = response.json()
    assert payload["total_embeddings"] == 2
    assert payload["dimensions"] == 3
    assert payload["embeddings"][0]["vector_preview"] == [0.6, 0.8, 0.0]
    assert payload["usage"]["input_tokens"] == 24
