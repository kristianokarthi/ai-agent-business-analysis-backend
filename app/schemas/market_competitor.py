from enum import Enum

from pydantic import Field, HttpUrl, model_validator

from app.schemas.business_fundamentals import (
    BusinessFundamentalsOutput,
    ConfidenceLevel,
)
from app.schemas.fact_finder import FactFinderOutput, StrictSchema
from app.schemas.llm import LLMUsage
from app.schemas.research import ResearchPurpose


class MarketAnalysisStatus(str, Enum):
    COMPLETED = "completed"
    INSUFFICIENT_DATA = "insufficient_data"


class MarketSourceType(str, Enum):
    COMPANY_OFFICIAL = "company_official"
    COMPETITOR_OFFICIAL = "competitor_official"
    GOVERNMENT = "government"
    INDUSTRY_REPORT = "industry_report"
    NEWS = "news"
    OTHER = "other"


class CompetitorType(str, Enum):
    DIRECT = "direct"
    INDIRECT = "indirect"


class MarketStructureType(str, Enum):
    CONCENTRATED = "concentrated"
    FRAGMENTED = "fragmented"
    EMERGING = "emerging"
    UNKNOWN = "unknown"


class MarketSourceDocument(StrictSchema):
    source_id: str = Field(
        min_length=1,
        pattern=r"^market_source_[A-Za-z0-9_-]+$",
    )
    title: str = Field(min_length=1)
    url: HttpUrl
    source_type: MarketSourceType
    content: str = Field(min_length=1, max_length=12000)


class MarketCompetitorInput(StrictSchema):
    purpose: ResearchPurpose
    follow_up_answers: dict[str, str] = Field(default_factory=dict)
    company_evidence: FactFinderOutput
    business_fundamentals: BusinessFundamentalsOutput
    market_documents: list[MarketSourceDocument] = Field(
        min_length=1,
        max_length=8,
    )

    @model_validator(mode="after")
    def validate_input_evidence(self) -> "MarketCompetitorInput":
        company_name = (
            self.company_evidence.company_identity.name
            .strip()
            .casefold()
        )
        analyzed_name = (
            self.business_fundamentals.company_name
            .strip()
            .casefold()
        )
        if company_name != analyzed_name:
            raise ValueError(
                "Agent 1 and Agent 2 company names must match."
            )

        evidence_ids = [
            *(fact.fact_id for fact in self.company_evidence.verified_facts),
            *(claim.claim_id for claim in self.company_evidence.company_claims),
            *(
                finding.finding_id
                for finding in self.business_fundamentals.findings
            ),
            *(document.source_id for document in self.market_documents),
        ]
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError(
                "All input evidence IDs must be unique."
            )

        return self


class MarketDefinition(StrictSchema):
    industry: str = Field(min_length=1)
    geographic_market: str = Field(min_length=1)
    customer_groups: list[str] = Field(max_length=5)
    evidence_ids: list[str] = Field(min_length=1)
    confidence: ConfidenceLevel


class CompetitorProfile(StrictSchema):
    competitor_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    competitor_type: CompetitorType
    positioning: str = Field(min_length=1)
    strengths: list[str] = Field(max_length=3)
    weaknesses: list[str] = Field(max_length=3)
    evidence_ids: list[str] = Field(min_length=1)
    confidence: ConfidenceLevel


class MarketStructure(StrictSchema):
    classification: MarketStructureType
    explanation: str = Field(min_length=1)
    evidence_ids: list[str] = Field(min_length=1)
    confidence: ConfidenceLevel


class MarketFinding(StrictSchema):
    finding_id: str = Field(min_length=1)
    statement: str = Field(min_length=1)
    evidence_ids: list[str] = Field(min_length=1)
    confidence: ConfidenceLevel


class MarketCompetitorOutput(StrictSchema):
    status: MarketAnalysisStatus
    company_name: str = Field(min_length=1)
    market_definition: MarketDefinition
    competitors: list[CompetitorProfile] = Field(max_length=5)
    market_structure: MarketStructure
    entry_barriers: list[MarketFinding] = Field(max_length=5)
    market_trends: list[MarketFinding] = Field(max_length=5)
    market_gaps: list[MarketFinding] = Field(max_length=5)
    competitive_risks: list[MarketFinding] = Field(max_length=5)
    missing_information: list[str]
    overall_confidence: ConfidenceLevel

    @model_validator(mode="after")
    def prevent_duplicate_items(self) -> "MarketCompetitorOutput":
        competitor_ids = [
            competitor.competitor_id
            for competitor in self.competitors
        ]
        competitor_names = [
            competitor.name.strip().casefold()
            for competitor in self.competitors
        ]
        findings = [
            *self.entry_barriers,
            *self.market_trends,
            *self.market_gaps,
            *self.competitive_risks,
        ]
        finding_ids = [finding.finding_id for finding in findings]
        statements = [
            finding.statement.strip().casefold()
            for finding in findings
        ]

        if len(competitor_ids) != len(set(competitor_ids)):
            raise ValueError("Competitor IDs must be unique.")
        if len(competitor_names) != len(set(competitor_names)):
            raise ValueError("Competitor names must be unique.")
        if len(finding_ids) != len(set(finding_ids)):
            raise ValueError("Market finding IDs must be unique.")
        if len(statements) != len(set(statements)):
            raise ValueError("Market findings must not be duplicated.")

        return self


class MarketCompetitorAPIResponse(StrictSchema):
    result: MarketCompetitorOutput
    usage: LLMUsage
