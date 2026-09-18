from enum import Enum

from pydantic import Field, HttpUrl

from app.schemas.business_fundamentals import ConfidenceLevel
from app.schemas.fact_finder import StrictSchema
from app.schemas.llm import LLMUsage
from app.schemas.report_context import (
    ReportContext,
    ReportContextRequest,
    ReportSourceType,
)
from app.schemas.research import ResearchPurpose


class ReportStatus(str, Enum):
    COMPLETED = "completed"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class ReportInsight(StrictSchema):
    title: str = Field(min_length=1, max_length=160)
    analysis: str = Field(min_length=1, max_length=2400)
    evidence_ids: list[str] = Field(min_length=1, max_length=8)
    confidence: ConfidenceLevel


class PurposeSpecificAnalysis(StrictSchema):
    focus: str = Field(min_length=1, max_length=240)
    key_considerations: list[ReportInsight] = Field(min_length=1, max_length=5)
    favorable_case: list[ReportInsight] = Field(max_length=4)
    caution_case: list[ReportInsight] = Field(max_length=4)
    decision_factors: list[ReportInsight] = Field(min_length=1, max_length=5)


class ReportConfidenceAssessment(StrictSchema):
    overall_confidence: ConfidenceLevel
    rationale: str = Field(min_length=1, max_length=1200)
    limitations: list[str] = Field(min_length=1, max_length=30)


class ReportStrategistDraft(StrictSchema):
    status: ReportStatus
    company_name: str = Field(min_length=1)
    purpose: ResearchPurpose
    title: str = Field(min_length=1, max_length=200)
    executive_summary: ReportInsight
    important_findings: list[ReportInsight] = Field(min_length=1, max_length=6)
    opportunities: list[ReportInsight] = Field(max_length=5)
    risks: list[ReportInsight] = Field(min_length=1, max_length=5)
    purpose_specific_analysis: PurposeSpecificAnalysis
    balanced_conclusion: ReportInsight
    questions_for_further_investigation: list[str] = Field(
        min_length=1,
        max_length=8,
    )
    confidence_assessment: ReportConfidenceAssessment
    missing_information: list[str] = Field(max_length=30)


class ReportCitationSource(StrictSchema):
    evidence_id: str
    title: str
    url: HttpUrl | None
    source_type: ReportSourceType


class StrategicReport(ReportStrategistDraft):
    sources_used: list[ReportCitationSource]
    word_count: int = Field(ge=1)
    quality_warnings: list[str]


class ReportContextMetrics(StrictSchema):
    estimated_tokens: int = Field(ge=0)
    token_budget: int = Field(ge=3_500, le=6_000)
    truncated_for_budget: bool
    evidence_source_count: int = Field(ge=0)


class ReportStrategistAPIResponse(StrictSchema):
    result: StrategicReport
    usage: LLMUsage
    context: ReportContextMetrics


class ReportStrategistResearchRequest(ReportContextRequest):
    pass


class ReportStrategistContextRequest(StrictSchema):
    context: ReportContext
