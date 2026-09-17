import pytest
from pydantic import ValidationError

from app.reports.context_builder import build_report_context
from app.schemas.report_context import ReportContextRequest
from app.tests.test_customer_reputation_schema import (
    VALID_OUTPUT as VALID_REPUTATION_OUTPUT,
)
from app.tests.test_market_competitor_schema import (
    VALID_BUSINESS_FUNDAMENTALS,
    VALID_COMPANY_EVIDENCE,
    VALID_OUTPUT as VALID_MARKET_OUTPUT,
)


def valid_request_data() -> dict:
    return {
        "purpose": "stock_research",
        "follow_up_answers": {
            "geographic_market": "India",
            "time_horizon": "long_term",
        },
        "company_evidence": VALID_COMPANY_EVIDENCE,
        "business_fundamentals": VALID_BUSINESS_FUNDAMENTALS,
        "market_analysis": VALID_MARKET_OUTPUT,
        "customer_reputation": VALID_REPUTATION_OUTPUT,
        "market_sources": [
            {
                "evidence_id": "market_source_1",
                "title": "PepsiCo brands",
                "url": "https://www.pepsico.com/our-brands",
                "source_type": "competitor_official",
            },
            {
                "evidence_id": "market_source_2",
                "title": "Beverage market overview",
                "url": "https://www.example.com/beverage-market",
                "source_type": "industry_report",
            },
        ],
        "public_signal_sources": [
            {
                "evidence_id": "signal_review_1",
                "title": "Customer review one",
                "url": "https://example.com/reviews/1",
                "source_type": "customer_review",
            },
            {
                "evidence_id": "signal_review_2",
                "title": "Customer review two",
                "url": "https://example.com/reviews/2",
                "source_type": "customer_review",
            },
            {
                "evidence_id": "signal_review_3",
                "title": "Customer review three",
                "url": "https://example.com/reviews/3",
                "source_type": "customer_review",
            },
            {
                "evidence_id": "signal_news_1",
                "title": "Packaging initiative report",
                "url": "https://example.com/news/packaging",
                "source_type": "news",
            },
        ],
        "collection_warnings": [
            "The public-signal sample is not population representative."
        ],
        "max_context_tokens": 5_000,
    }


def test_builder_creates_compact_context_and_terminal_citations():
    request = ReportContextRequest.model_validate(valid_request_data())

    result = build_report_context(request)

    assert result.company_name == "The Coca-Cola Company"
    assert result.purpose.value == "stock_research"
    assert result.estimated_tokens <= result.token_budget
    assert result.truncated_for_budget is False
    assert result.business_findings[0].evidence_ids == ["source_1"]
    assert result.market.competitors[0].evidence_ids == [
        "market_source_1"
    ]
    assert result.reputation.praise[0].evidence_ids == [
        "signal_review_1",
        "signal_review_3",
    ]

    catalog = {
        source.evidence_id: source
        for source in result.evidence_catalog
    }
    assert catalog["source_1"].url is not None
    assert catalog["market_source_1"].source_type.value == (
        "competitor_official"
    )
    assert catalog["signal_review_1"].source_type.value == (
        "customer_review"
    )


def test_builder_preserves_all_missing_information_and_warnings():
    request = ReportContextRequest.model_validate(valid_request_data())

    result = build_report_context(request)

    assert "Market share" in result.missing_information
    assert "Geographic market definition" in result.missing_information
    assert (
        "The four-signal sample is too small for population conclusions."
        in result.missing_information
    )
    assert (
        "The public-signal sample is not population representative."
        in result.missing_information
    )
    assert any(
        "public sample of 4 signals" in warning
        for warning in result.missing_information
    )


def test_builder_excludes_raw_source_content_from_context():
    request = ReportContextRequest.model_validate(valid_request_data())

    result = build_report_context(request)
    serialized = result.model_dump_json()

    assert "I like the taste and the product is easy to find" not in serialized
    assert "The beverage market includes multiple brands" not in serialized
    assert "Customer review one" in serialized


def test_request_rejects_mismatched_agent_company_names():
    data = valid_request_data()
    data["customer_reputation"] = {
        **VALID_REPUTATION_OUTPUT,
        "company_name": "Another Company",
    }

    with pytest.raises(
        ValidationError,
        match="Agents 1–4 company names must match",
    ):
        ReportContextRequest.model_validate(data)


def test_builder_trims_optional_findings_to_respect_small_budget():
    data = valid_request_data()
    long_statement = (
        "This is an intentionally detailed evidence-backed business finding "
        "used to exercise deterministic report-context budget trimming. " * 8
    )
    data["company_evidence"] = {
        **VALID_COMPANY_EVIDENCE,
        "verified_facts": [
            {
                "fact_id": f"fact_{index}",
                "statement": f"{long_statement} Fact {index}.",
                "category": "financial",
                "source_ids": ["source_1"],
            }
            for index in range(1, 9)
        ],
    }
    data["business_fundamentals"] = {
        **VALID_BUSINESS_FUNDAMENTALS,
        "findings": [
            {
                "finding_id": f"business_finding_{index}",
                "area": "financial_highlights",
                "statement": f"{long_statement} Finding {index}.",
                "basis": "verified_fact",
                "evidence_ids": [f"fact_{min(index, 8)}"],
                "confidence": "high",
            }
            for index in range(1, 13)
        ],
    }
    data["market_analysis"] = {
        **VALID_MARKET_OUTPUT,
        "market_definition": {
            **VALID_MARKET_OUTPUT["market_definition"],
            "evidence_ids": ["fact_1"],
        },
        "entry_barriers": [
            {
                **VALID_MARKET_OUTPUT["entry_barriers"][0],
                "evidence_ids": ["fact_1", "market_source_1"],
            }
        ],
    }
    data["max_context_tokens"] = 3_500

    result = build_report_context(
        ReportContextRequest.model_validate(data)
    )

    assert result.estimated_tokens <= 3_500
    assert result.truncated_for_budget is True
    assert len(result.business_findings) < 8
    assert len(result.verified_facts) >= 4

