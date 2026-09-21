from enum import Enum

from pydantic import Field

from app.schemas.fact_finder import StrictSchema
from app.schemas.llm import LLMUsage
from app.schemas.report_strategist import (
    ReportCitationSource,
    StrategicReport,
)
from app.schemas.research import ResearchPurpose


class ReportChunkSection(str, Enum):
    EXECUTIVE_SUMMARY = "executive_summary"
    IMPORTANT_FINDING = "important_finding"
    OPPORTUNITY = "opportunity"
    RISK = "risk"
    KEY_CONSIDERATION = "key_consideration"
    FAVORABLE_CASE = "favorable_case"
    CAUTION_CASE = "caution_case"
    DECISION_FACTOR = "decision_factor"
    BALANCED_CONCLUSION = "balanced_conclusion"
    INVESTIGATION_QUESTION = "investigation_question"
    LIMITATION = "limitation"
    MISSING_INFORMATION = "missing_information"
    QUALITY_WARNING = "quality_warning"


class ReportChunk(StrictSchema):
    chunk_id: str = Field(min_length=1, max_length=100)
    section: ReportChunkSection
    title: str = Field(min_length=1, max_length=200)
    content: str = Field(min_length=1, max_length=3000)
    evidence_ids: list[str] = Field(default_factory=list, max_length=8)
    sources: list[ReportCitationSource] = Field(
        default_factory=list,
        max_length=8,
    )


class ReportChunkPreviewRequest(StrictSchema):
    report: StrategicReport


class ReportChunkPreviewResponse(StrictSchema):
    company_name: str
    purpose: ResearchPurpose
    total_chunks: int = Field(ge=1)
    section_counts: dict[ReportChunkSection, int]
    chunks: list[ReportChunk] = Field(min_length=1, max_length=120)


class EmbeddingPreviewRequest(StrictSchema):
    chunks: list[ReportChunk] = Field(min_length=1, max_length=50)


class ChunkEmbeddingPreview(StrictSchema):
    chunk_id: str
    section: ReportChunkSection
    title: str
    dimensions: int = Field(ge=1)
    vector_preview: list[float] = Field(min_length=1, max_length=8)


class EmbeddingUsage(StrictSchema):
    provider: str
    model: str
    input_tokens: int = Field(ge=0)
    requests: int = Field(ge=1)


class EmbeddingPreviewResponse(StrictSchema):
    total_embeddings: int = Field(ge=1)
    dimensions: int = Field(ge=1)
    embeddings: list[ChunkEmbeddingPreview] = Field(
        min_length=1,
        max_length=50,
    )
    usage: EmbeddingUsage


class SemanticSearchRequest(StrictSchema):
    question: str = Field(min_length=3, max_length=500)
    chunks: list[ReportChunk] = Field(min_length=1, max_length=50)
    top_k: int = Field(default=3, ge=1, le=10)


class SemanticSearchMatch(StrictSchema):
    rank: int = Field(ge=1)
    similarity_score: float = Field(ge=-1, le=1)
    chunk: ReportChunk


class SemanticSearchResponse(StrictSchema):
    question: str
    total_chunks_searched: int = Field(ge=1)
    matches_returned: int = Field(ge=1)
    matches: list[SemanticSearchMatch] = Field(
        min_length=1,
        max_length=10,
    )
    usage: EmbeddingUsage


class GroundedAnswerStatus(str, Enum):
    ANSWERED = "answered"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    OUT_OF_SCOPE = "out_of_scope"


class ChatRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"


class ChatMessage(StrictSchema):
    role: ChatRole
    content: str = Field(min_length=1, max_length=3000)


class GroundedAnswerDraft(StrictSchema):
    status: GroundedAnswerStatus
    answer: str = Field(min_length=1, max_length=3000)
    supporting_chunk_ids: list[str] = Field(max_length=5)
    limitations: list[str] = Field(max_length=3)


class GroundedAnswerRequest(SemanticSearchRequest):
    conversation_history: list[ChatMessage] = Field(
        default_factory=list,
        max_length=6,
    )


class GroundedAnswerResponse(StrictSchema):
    status: GroundedAnswerStatus
    question: str
    answer: str
    supporting_chunks: list[SemanticSearchMatch] = Field(max_length=5)
    evidence_ids: list[str] = Field(max_length=40)
    sources: list[ReportCitationSource] = Field(max_length=40)
    limitations: list[str] = Field(max_length=3)
    retrieval_usage: EmbeddingUsage
    generation_usage: LLMUsage | None
