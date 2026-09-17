from enum import Enum

from pydantic import Field, HttpUrl, model_validator

from app.schemas.business_fundamentals import (
    BusinessFundamentalsOutput,
    ConfidenceLevel,
)
from app.schemas.customer_reputation import (
    CustomerReputationOutput,
    SignalSentiment,
)
from app.schemas.fact_finder import CompanyIdentity, FactFinderOutput, StrictSchema
from app.schemas.market_competitor import MarketCompetitorOutput
from app.schemas.research import ResearchPurpose


class ReportSourceType(str, Enum):
    COMPANY_OFFICIAL = "company_official"
    COMPETITOR_OFFICIAL = "competitor_official"
    GOVERNMENT = "government"
    INDUSTRY_REPORT = "industry_report"
    NEWS = "news"
    CUSTOMER_REVIEW = "customer_review"
    EMPLOYEE_REVIEW = "employee_review"
    FORUM = "forum"
    SOCIAL_MEDIA = "social_media"
    SURVEY = "survey"
    OTHER = "other"
    UNKNOWN = "unknown"


class ReportSourceReference(StrictSchema):
    evidence_id: str = Field(min_length=1)
    title: str = Field(min_length=1, max_length=300)
    url: HttpUrl | None = None
    source_type: ReportSourceType


class ReportContextRequest(StrictSchema):
    purpose: ResearchPurpose
    follow_up_answers: dict[str, str] = Field(default_factory=dict)
    company_evidence: FactFinderOutput
    business_fundamentals: BusinessFundamentalsOutput
    market_analysis: MarketCompetitorOutput
    customer_reputation: CustomerReputationOutput
    company_sources: list[ReportSourceReference] = Field(
        default_factory=list,
        max_length=5,
    )
    market_sources: list[ReportSourceReference] = Field(
        default_factory=list,
        max_length=8,
    )
    public_signal_sources: list[ReportSourceReference] = Field(
        default_factory=list,
        max_length=20,
    )
    collection_warnings: list[str] = Field(default_factory=list, max_length=20)
    max_context_tokens: int = Field(default=5_000, ge=3_500, le=6_000)

    @model_validator(mode="after")
    def validate_agent_identity_and_source_ids(self) -> "ReportContextRequest":
        expected = self.company_evidence.company_identity.name.strip().casefold()
        analyzed_names = (
            self.business_fundamentals.company_name,
            self.market_analysis.company_name,
            self.customer_reputation.company_name,
        )
        if any(name.strip().casefold() != expected for name in analyzed_names):
            raise ValueError("Agents 1–4 company names must match.")

        source_ids = [
            source.evidence_id
            for source in (
                *self.company_sources,
                *self.market_sources,
                *self.public_signal_sources,
            )
        ]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("Report source evidence IDs must be unique.")
        return self


class ContextFinding(StrictSchema):
    finding_id: str = Field(min_length=1)
    statement: str = Field(min_length=1)
    category: str = Field(min_length=1)
    evidence_ids: list[str]
    confidence: ConfidenceLevel


class ContextConflict(StrictSchema):
    topic: str = Field(min_length=1)
    statements: list[str] = Field(min_length=2)
    evidence_ids: list[str]


class ContextCompetitor(StrictSchema):
    name: str = Field(min_length=1)
    competitor_type: str = Field(min_length=1)
    positioning: str = Field(min_length=1)
    strengths: list[str] = Field(max_length=3)
    weaknesses: list[str] = Field(max_length=3)
    evidence_ids: list[str]
    confidence: ConfidenceLevel


class ContextMarketSummary(StrictSchema):
    industry: str
    geographic_market: str
    customer_groups: list[str]
    market_structure: str
    structure_explanation: str
    evidence_ids: list[str]
    competitors: list[ContextCompetitor] = Field(max_length=3)
    entry_barriers: list[ContextFinding] = Field(max_length=3)
    market_trends: list[ContextFinding] = Field(max_length=3)
    market_gaps: list[ContextFinding] = Field(max_length=3)
    competitive_risks: list[ContextFinding] = Field(max_length=3)
    confidence: ConfidenceLevel


class ContextTheme(StrictSchema):
    statement: str = Field(min_length=1)
    evidence_ids: list[str]
    confidence: ConfidenceLevel


class ContextReputationSummary(StrictSchema):
    sample_size: int = Field(ge=0)
    sentiment: SignalSentiment
    positive_count: int = Field(ge=0)
    neutral_count: int = Field(ge=0)
    negative_count: int = Field(ge=0)
    mixed_count: int = Field(ge=0)
    unclear_count: int = Field(ge=0)
    praise: list[ContextTheme] = Field(max_length=3)
    complaints: list[ContextTheme] = Field(max_length=3)
    pain_points: list[ContextTheme] = Field(max_length=3)
    unmet_needs: list[ContextTheme] = Field(max_length=3)
    reputation_risks: list[ContextTheme] = Field(max_length=3)
    conflicting_signals: list[ContextTheme] = Field(max_length=3)
    confidence: ConfidenceLevel


class EvidenceCatalogEntry(ReportSourceReference):
    used_by: list[str]


class ReportContext(StrictSchema):
    company_name: str = Field(min_length=1)
    purpose: ResearchPurpose
    follow_up_answers: dict[str, str]
    company_identity: CompanyIdentity
    identity_evidence_ids: list[str]
    verified_facts: list[ContextFinding] = Field(max_length=6)
    company_claims: list[ContextFinding] = Field(max_length=3)
    conflicting_claims: list[ContextConflict] = Field(max_length=3)
    business_findings: list[ContextFinding] = Field(max_length=8)
    market: ContextMarketSummary
    reputation: ContextReputationSummary
    missing_information: list[str]
    evidence_catalog: list[EvidenceCatalogEntry]
    estimated_tokens: int = Field(ge=0)
    token_budget: int = Field(ge=3_500, le=6_000)
    truncated_for_budget: bool
