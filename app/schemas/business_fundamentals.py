from enum import Enum

from pydantic import Field, model_validator

from app.schemas.fact_finder import (
    FactFinderOutput,
    StrictSchema,
)
from app.schemas.llm import LLMUsage
from app.schemas.research import ResearchPurpose


class BusinessAnalysisStatus(str, Enum):
    COMPLETED = "completed"
    INSUFFICIENT_DATA = "insufficient_data"


class ConfidenceLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class FindingBasis(str, Enum):
    VERIFIED_FACT = "verified_fact"
    COMPANY_CLAIM = "company_claim"
    REASONED_INFERENCE = "reasoned_inference"


class BusinessArea(str, Enum):
    BUSINESS_MODEL = "business_model"
    PRODUCTS_SERVICES = "products_services"
    CUSTOMER_SEGMENTS = "customer_segments"
    VALUE_PROPOSITION = "value_proposition"
    REVENUE_SOURCES = "revenue_sources"
    SALES_DISTRIBUTION = "sales_distribution"
    COST_DRIVERS = "cost_drivers"
    RESOURCES_CAPABILITIES = "resources_capabilities"
    OPERATIONAL_STRENGTHS = "operational_strengths"
    OPERATIONAL_WEAKNESSES = "operational_weaknesses"
    FINANCIAL_HIGHLIGHTS = "financial_highlights"


class BusinessFundamentalsInput(StrictSchema):
    purpose: ResearchPurpose
    follow_up_answers: dict[str, str] = Field(
        default_factory=dict,
    )
    evidence_pack: FactFinderOutput


class BusinessFinding(StrictSchema):
    finding_id: str = Field(min_length=1)
    area: BusinessArea
    statement: str = Field(min_length=1)
    basis: FindingBasis
    evidence_ids: list[str] = Field(min_length=1)
    confidence: ConfidenceLevel


class BusinessFundamentalsOutput(StrictSchema):
    status: BusinessAnalysisStatus
    company_name: str = Field(min_length=1)
    findings: list[BusinessFinding] = Field(max_length=15)
    missing_information: list[str]
    overall_confidence: ConfidenceLevel

    @model_validator(mode="after")
    def prevent_duplicate_findings(
        self,
    ) -> "BusinessFundamentalsOutput":
        finding_ids = [
            finding.finding_id
            for finding in self.findings
        ]
        if len(finding_ids) != len(set(finding_ids)):
            raise ValueError("Finding IDs must be unique.")

        statements = [
            finding.statement.strip().lower()
            for finding in self.findings
        ]
        if len(statements) != len(set(statements)):
            raise ValueError(
                "Business findings must not be duplicated."
            )

        return self


class BusinessFundamentalsAPIResponse(StrictSchema):
    result: BusinessFundamentalsOutput
    usage: LLMUsage
