import pytest
from pydantic import ValidationError

from app.schemas.market_competitor import (
    MarketCompetitorInput,
    MarketCompetitorOutput,
)


VALID_COMPANY_EVIDENCE = {
    "status": "completed",
    "company_identity": {
        "name": "The Coca-Cola Company",
        "legal_name": "The Coca-Cola Company",
        "industry": "Beverages",
        "headquarters": "Atlanta, Georgia",
        "official_website": "https://www.coca-colacompany.com",
    },
    "verified_facts": [
        {
            "fact_id": "fact_1",
            "statement": "The company sells branded beverages.",
            "category": "products_services",
            "source_ids": ["source_1"],
        }
    ],
    "company_claims": [],
    "conflicting_claims": [],
    "missing_information": [],
    "sources_used": ["source_1"],
}


VALID_BUSINESS_FUNDAMENTALS = {
    "status": "completed",
    "company_name": "The Coca-Cola Company",
    "findings": [
        {
            "finding_id": "business_finding_1",
            "area": "products_services",
            "statement": "The company sells branded beverages.",
            "basis": "verified_fact",
            "evidence_ids": ["fact_1"],
            "confidence": "high",
        }
    ],
    "missing_information": ["Market share"],
    "overall_confidence": "medium",
}


VALID_INPUT = {
    "purpose": "start_similar_business",
    "follow_up_answers": {
        "target_location": "Chennai",
    },
    "company_evidence": VALID_COMPANY_EVIDENCE,
    "business_fundamentals": VALID_BUSINESS_FUNDAMENTALS,
    "market_documents": [
        {
            "source_id": "market_source_1",
            "title": "Competitor information",
            "url": "https://www.pepsico.com/our-brands",
            "source_type": "competitor_official",
            "content": "PepsiCo offers a portfolio of beverage brands.",
        },
        {
            "source_id": "market_source_2",
            "title": "Industry information",
            "url": "https://www.example.com/beverage-market",
            "source_type": "industry_report",
            "content": "The beverage market includes multiple brands.",
        },
    ],
}


VALID_OUTPUT = {
    "status": "completed",
    "company_name": "The Coca-Cola Company",
    "market_definition": {
        "industry": "Beverages",
        "geographic_market": "Not established by the supplied evidence",
        "customer_groups": [],
        "evidence_ids": ["fact_1"],
        "confidence": "low",
    },
    "competitors": [
        {
            "competitor_id": "competitor_1",
            "name": "PepsiCo",
            "competitor_type": "direct",
            "positioning": "Offers a portfolio of beverage brands.",
            "strengths": ["Portfolio of beverage brands"],
            "weaknesses": [],
            "evidence_ids": ["market_source_1"],
            "confidence": "medium",
        }
    ],
    "market_structure": {
        "classification": "unknown",
        "explanation": (
            "The supplied evidence does not establish market concentration."
        ),
        "evidence_ids": ["market_source_2"],
        "confidence": "low",
    },
    "entry_barriers": [
        {
            "finding_id": "barrier_1",
            "statement": (
                "A new entrant would compete with established "
                "beverage portfolios."
            ),
            "evidence_ids": ["fact_1", "market_source_1"],
            "confidence": "low",
        }
    ],
    "market_trends": [],
    "market_gaps": [],
    "competitive_risks": [],
    "missing_information": [
        "Geographic market definition",
        "Independent market-share information",
    ],
    "overall_confidence": "low",
}


def test_market_competitor_accepts_valid_input():
    result = MarketCompetitorInput.model_validate(VALID_INPUT)

    assert result.purpose.value == "start_similar_business"
    assert len(result.market_documents) == 2


def test_market_competitor_rejects_mismatched_companies():
    invalid_input = {
        **VALID_INPUT,
        "business_fundamentals": {
            **VALID_BUSINESS_FUNDAMENTALS,
            "company_name": "Another Company",
        },
    }

    with pytest.raises(
        ValidationError,
        match="company names must match",
    ):
        MarketCompetitorInput.model_validate(invalid_input)


def test_market_competitor_rejects_duplicate_source_ids():
    duplicate_document = {
        **VALID_INPUT["market_documents"][1],
        "source_id": "market_source_1",
    }
    invalid_input = {
        **VALID_INPUT,
        "market_documents": [
            VALID_INPUT["market_documents"][0],
            duplicate_document,
        ],
    }

    with pytest.raises(
        ValidationError,
        match="evidence IDs must be unique",
    ):
        MarketCompetitorInput.model_validate(invalid_input)


def test_market_competitor_accepts_valid_output():
    result = MarketCompetitorOutput.model_validate(VALID_OUTPUT)

    assert result.status.value == "completed"
    assert result.competitors[0].name == "PepsiCo"
    assert result.market_structure.classification.value == "unknown"


def test_market_competitor_rejects_duplicate_findings():
    duplicate_finding = {
        **VALID_OUTPUT["entry_barriers"][0],
        "finding_id": "trend_1",
    }
    invalid_output = {
        **VALID_OUTPUT,
        "market_trends": [duplicate_finding],
    }

    with pytest.raises(
        ValidationError,
        match="must not be duplicated",
    ):
        MarketCompetitorOutput.model_validate(invalid_output)
