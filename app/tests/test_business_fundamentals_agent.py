import pytest

from app.agents.business_fundamentals import (
    InvalidEvidenceReferenceError,
    validate_evidence_references,
)
from app.schemas.business_fundamentals import (
    BusinessFundamentalsOutput,
)
from app.schemas.fact_finder import FactFinderOutput
from app.tests.test_business_fundamentals_schema import (
    VALID_EVIDENCE_PACK,
    VALID_OUTPUT,
)


def test_evidence_validator_accepts_known_fact_and_claim_ids():
    evidence_pack = FactFinderOutput.model_validate(
        VALID_EVIDENCE_PACK,
    )
    output_data = {
        **VALID_OUTPUT,
        "findings": [
            VALID_OUTPUT["findings"][0],
            {
                "finding_id": "finding_2",
                "area": "customer_segments",
                "statement": "The company reports a global market reach.",
                "basis": "company_claim",
                "evidence_ids": ["claim_1"],
                "confidence": "medium",
            },
        ],
    }
    output = BusinessFundamentalsOutput.model_validate(output_data)

    validate_evidence_references(output, evidence_pack)


def test_evidence_validator_rejects_unknown_evidence_id():
    evidence_pack = FactFinderOutput.model_validate(
        VALID_EVIDENCE_PACK,
    )
    output_data = {
        **VALID_OUTPUT,
        "findings": [
            {
                **VALID_OUTPUT["findings"][0],
                "evidence_ids": ["fact_999"],
            }
        ],
    }
    output = BusinessFundamentalsOutput.model_validate(output_data)

    with pytest.raises(
        InvalidEvidenceReferenceError,
        match="unknown evidence",
    ):
        validate_evidence_references(output, evidence_pack)


def test_evidence_validator_rejects_claim_marked_as_verified():
    evidence_pack = FactFinderOutput.model_validate(
        VALID_EVIDENCE_PACK,
    )
    output_data = {
        **VALID_OUTPUT,
        "findings": [
            {
                **VALID_OUTPUT["findings"][0],
                "basis": "verified_fact",
                "evidence_ids": ["claim_1"],
            }
        ],
    }
    output = BusinessFundamentalsOutput.model_validate(output_data)

    with pytest.raises(
        InvalidEvidenceReferenceError,
        match="cites a company claim",
    ):
        validate_evidence_references(output, evidence_pack)
