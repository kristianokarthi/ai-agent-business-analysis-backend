from enum import Enum

from pydantic import Field

from app.schemas.fact_finder import StrictSchema
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
