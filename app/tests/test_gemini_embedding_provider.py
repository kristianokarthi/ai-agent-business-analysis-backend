import json
import math

import httpx
import pytest
from fastapi.testclient import TestClient

from app.api.rag import get_embedding_provider
from app.embeddings.gemini_provider import (
    GeminiEmbeddingConfigurationError,
    GeminiEmbeddingProvider,
    GeminiEmbeddingRateLimitError,
)
from app.main import app
from app.schemas.rag import ReportChunk, ReportChunkSection


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
