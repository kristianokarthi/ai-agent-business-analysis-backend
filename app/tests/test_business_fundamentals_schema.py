import pytest
from pydantic import ValidationError

from app.schemas.business_fundamentals import (
    BusinessFundamentalsInput,
    BusinessFundamentalsOutput,
)


VALID_EVIDENCE_PACK = {
    "status": "completed",
    "company_identity": {
        "name": "The Coca-Cola Company",
        "legal_name": "The Coca-Cola Company",
        "industry": "Beverages",
        "headquarters": "Atlanta, United States",
        "official_website": "https://www.coca-colacompany.com",
    },
    "verified_facts": [
        {
            "fact_id": "fact_1",
            "statement": "The company sells branded beverages.",
            "category": "products_services",
            "source_ids": ["source_1"],
        },
        {
            "fact_id": "fact_2",
            "statement": "Bottling partners distribute its beverages.",
            "category": "operations",
            "source_ids": ["source_1"],
        },
    ],
    "company_claims": [
        {
            "claim_id": "claim_1",
            "statement": "Its products are sold in over 200 countries.",
            "category": "customers_markets",
            "source_ids": ["source_1"],
        }
    ],
    "conflicting_claims": [],
    "missing_information": ["Detailed revenue data"],
    "sources_used": ["source_1"],
}


VALID_INPUT = {
    "purpose": "start_similar_business",
    "follow_up_answers": {
        "target_location": "Chennai",
        "investment_amount": "INR 5-20 lakh",
    },
    "evidence_pack": VALID_EVIDENCE_PACK,
}


VALID_OUTPUT = {
    "status": "completed",
    "company_name": "The Coca-Cola Company",
    "findings": [
        {
            "finding_id": "finding_1",
            "area": "products_services",
            "statement": "The company sells branded beverages.",
            "basis": "verified_fact",
            "evidence_ids": ["fact_1"],
            "confidence": "high",
        },
        {
            "finding_id": "finding_2",
            "area": "sales_distribution",
            "statement": "Bottling partners form a distribution channel.",
            "basis": "reasoned_inference",
            "evidence_ids": ["fact_2"],
            "confidence": "medium",
        },
    ],
    "missing_information": [
        "Revenue mix",
        "Cost structure",
    ],
    "overall_confidence": "medium",
}


def test_business_fundamentals_accepts_valid_input():
    result = BusinessFundamentalsInput.model_validate(VALID_INPUT)

    assert result.purpose.value == "start_similar_business"
    assert result.evidence_pack.company_identity.name == (
        "The Coca-Cola Company"
    )
    assert len(result.evidence_pack.verified_facts) == 2


def test_business_fundamentals_rejects_unknown_input_fields():
    invalid_input = {
        **VALID_INPUT,
        "documents": [],
    }

    with pytest.raises(ValidationError):
        BusinessFundamentalsInput.model_validate(invalid_input)


def test_business_fundamentals_accepts_valid_output():
    result = BusinessFundamentalsOutput.model_validate(VALID_OUTPUT)

    assert result.status.value == "completed"
    assert len(result.findings) == 2
    assert result.findings[0].evidence_ids == ["fact_1"]


def test_business_fundamentals_rejects_empty_evidence_ids():
    invalid_output = {
        **VALID_OUTPUT,
        "findings": [
            {
                **VALID_OUTPUT["findings"][0],
                "evidence_ids": [],
            }
        ],
    }

    with pytest.raises(ValidationError):
        BusinessFundamentalsOutput.model_validate(invalid_output)


def test_business_fundamentals_rejects_duplicate_findings():
    invalid_output = {
        **VALID_OUTPUT,
        "findings": [
            VALID_OUTPUT["findings"][0],
            {
                **VALID_OUTPUT["findings"][0],
                "finding_id": "finding_2",
            },
        ],
    }

    with pytest.raises(
        ValidationError,
        match="must not be duplicated",
    ):
        BusinessFundamentalsOutput.model_validate(invalid_output)
