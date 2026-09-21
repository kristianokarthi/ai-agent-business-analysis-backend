import httpx
import pytest
from fastapi.testclient import TestClient

from app.agents.rag_answer import (
    GroundedAnswerAgent,
    InvalidGroundedAnswerError,
)
from app.api.rag import get_embedding_provider, get_groq_provider
from app.embeddings.gemini_provider import GeminiEmbeddingProvider
from app.llm.types import StructuredLLMResult
from app.main import app
from app.schemas.llm import LLMUsage
from app.schemas.rag import (
    GroundedAnswerDraft,
    GroundedAnswerStatus,
    ReportChunk,
    ReportChunkSection,
    SemanticSearchMatch,
)
from app.schemas.report_strategist import ReportCitationSource


def matches() -> list[SemanticSearchMatch]:
    return [
        SemanticSearchMatch(
            rank=1,
            similarity_score=0.72,
            chunk=ReportChunk(
                chunk_id="risk_02",
                section=ReportChunkSection.RISK,
                title="Post-Sales Service Concerns",
                content="Customers reported inadequate post-sales support.",
                evidence_ids=["signal_review_1"],
                sources=[
                    ReportCitationSource(
                        evidence_id="signal_review_1",
                        title="Customer feedback sample",
                        url="https://example.com/customer-feedback",
                        source_type="customer_review",
                    )
                ],
            ),
        )
    ]


class FakeLLMProvider:
    def __init__(self, supporting_chunk_ids: list[str]) -> None:
        self.supporting_chunk_ids = supporting_chunk_ids

    async def generate_structured(self, **kwargs):
        return StructuredLLMResult(
            data=GroundedAnswerDraft(
                status=GroundedAnswerStatus.ANSWERED,
                answer=(
                    "The supplied evidence identifies inadequate "
                    "post-sales support as a customer-service concern."
                ),
                supporting_chunk_ids=self.supporting_chunk_ids,
                limitations=["The evidence is based on a public sample."],
            ),
            usage=LLMUsage(
                agent_name="rag_grounded_answer",
                provider="groq",
                model="test-model",
                input_tokens=100,
                output_tokens=40,
                total_tokens=140,
            ),
        )


@pytest.mark.anyio
async def test_grounded_answer_accepts_retrieved_chunk_reference():
    result = await GroundedAnswerAgent(
        FakeLLMProvider(["risk_02"])
    ).run("What customer-service issue was reported?", matches())

    assert result.data.supporting_chunk_ids == ["risk_02"]


@pytest.mark.anyio
async def test_grounded_answer_rejects_unsupported_chunk_reference():
    with pytest.raises(InvalidGroundedAnswerError):
        await GroundedAnswerAgent(
            FakeLLMProvider(["invented_chunk"])
        ).run("What customer-service issue was reported?", matches())


def test_grounded_answer_preview_endpoint_returns_validated_sources():
    call_count = 0

    async def embedding_handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return httpx.Response(
                200,
                json={
                    "embeddings": [{"values": [0, 2, 0]}],
                    "usageMetadata": {"promptTokenCount": 15},
                },
            )
        return httpx.Response(
            200,
            json={
                "embeddings": [{"values": [0, 3, 0]}],
                "usageMetadata": {"promptTokenCount": 5},
            },
        )

    embedding_provider = GeminiEmbeddingProvider(
        api_key="test-key",
        dimensions=3,
        transport=httpx.MockTransport(embedding_handler),
    )
    app.dependency_overrides[get_embedding_provider] = (
        lambda: embedding_provider
    )
    app.dependency_overrides[get_groq_provider] = (
        lambda: FakeLLMProvider(["risk_02"])
    )
    try:
        response = TestClient(app).post(
            "/api/rag/answer/preview",
            json={
                "question": "What customer-service issue was reported?",
                "chunks": [
                    match.chunk.model_dump(mode="json")
                    for match in matches()
                ],
                "top_k": 1,
            },
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "answered"
    assert payload["supporting_chunks"][0]["chunk"]["chunk_id"] == "risk_02"
    assert payload["evidence_ids"] == ["signal_review_1"]
    assert payload["sources"][0]["title"] == "Customer feedback sample"
    assert payload["retrieval_usage"]["requests"] == 2
    assert payload["generation_usage"]["provider"] == "groq"
