import asyncio

import pytest

from app.agents.report_strategist import (
    InvalidReportEvidenceError,
    ReportStrategistAgent,
    UnsafeStockRecommendationError,
    context_metrics,
    finalize_report,
    validate_report_draft,
)
from app.llm.types import StructuredLLMResult
from app.reports.context_builder import build_report_context
from app.schemas.llm import LLMUsage
from app.schemas.report_context import ReportContextRequest
from app.schemas.report_strategist import ReportStrategistDraft
from app.tests.test_report_context_builder import valid_request_data


def report_context():
    return build_report_context(
        ReportContextRequest.model_validate(valid_request_data())
    )


def insight(title: str, evidence_id: str = "source_1") -> dict:
    return {
        "title": title,
        "analysis": (
            "The supplied evidence supports this bounded conclusion while "
            "important financial and market details still require validation."
        ),
        "evidence_ids": [evidence_id],
        "confidence": "medium",
    }


def valid_draft_data() -> dict:
    return {
        "status": "completed",
        "company_name": "The Coca-Cola Company",
        "purpose": "stock_research",
        "title": "Evidence-based company research overview",
        "executive_summary": insight("Executive view"),
        "important_findings": [
            insight("Business model"),
            insight("Market position", "market_source_1"),
        ],
        "opportunities": [
            insight("Market opportunity", "market_source_2"),
        ],
        "risks": [
            insight("Evidence limitation", "market_source_1"),
        ],
        "purpose_specific_analysis": {
            "focus": "Long-term stock research without a trading instruction",
            "key_considerations": [insight("Operating consideration")],
            "favorable_case": [
                insight("Favorable case", "market_source_1")
            ],
            "caution_case": [
                insight("Caution case", "signal_review_2")
            ],
            "decision_factors": [
                insight("Decision factor", "market_source_2")
            ],
        },
        "balanced_conclusion": insight(
            "Balanced conclusion",
            "market_source_1",
        ),
        "questions_for_further_investigation": [
            "What are the current India-specific revenue and margin trends?"
        ],
        "confidence_assessment": {
            "overall_confidence": "medium",
            "rationale": "The core claims have citations, but material gaps remain.",
            "limitations": ["Current valuation evidence was not supplied."],
        },
        "missing_information": ["Current valuation was not established."],
    }


class FakeOpenRouterProvider:
    async def generate_structured(self, **kwargs):
        assert kwargs["agent_name"] == "report_strategist"
        assert kwargs["response_model"] is ReportStrategistDraft
        assert kwargs["max_tokens"] == 6000
        assert "Allowed citation evidence IDs" in kwargs["user_prompt"]
        assert "market_source_1" in kwargs["user_prompt"]
        assert "Do not cite finding_id values" in kwargs["user_prompt"]
        return StructuredLLMResult(
            data=ReportStrategistDraft.model_validate(valid_draft_data()),
            usage=LLMUsage(
                agent_name="report_strategist",
                provider="openrouter",
                model="nvidia/nemotron-3-super-120b-a12b",
                input_tokens=1800,
                output_tokens=1400,
                reasoning_tokens=100,
                total_tokens=3300,
            ),
        )


def test_agent_finalizes_sources_warnings_and_word_count():
    context = report_context()
    result = asyncio.run(
        ReportStrategistAgent(provider=FakeOpenRouterProvider()).run(context)
    )

    assert result.data.company_name == context.company_name
    assert result.data.word_count > 0
    assert any(
        item == "Market share"
        for item in result.data.missing_information
    )
    assert {source.evidence_id for source in result.data.sources_used} >= {
        "source_1",
        "market_source_1",
    }
    assert result.usage.provider == "openrouter"


def test_unknown_evidence_id_is_rejected():
    context = report_context()
    data = valid_draft_data()
    data["executive_summary"] = insight(
        "Unsupported citation",
        "invented_source",
    )
    draft = ReportStrategistDraft.model_validate(data)

    with pytest.raises(InvalidReportEvidenceError, match="unknown evidence"):
        validate_report_draft(draft, context)


def test_direct_stock_instruction_is_rejected():
    context = report_context()
    data = valid_draft_data()
    data["balanced_conclusion"] = {
        **insight("Instruction"),
        "analysis": "Based on this report, buy the stock immediately.",
    }
    draft = ReportStrategistDraft.model_validate(data)

    with pytest.raises(UnsafeStockRecommendationError):
        validate_report_draft(draft, context)


def test_finalizer_preserves_context_warnings_and_reports_metrics():
    context = report_context()
    draft = ReportStrategistDraft.model_validate(valid_draft_data())
    report = finalize_report(draft, context)
    metrics = context_metrics(context)

    assert set(context.missing_information).issubset(
        set(report.missing_information)
    )
    assert metrics.estimated_tokens == context.estimated_tokens
    assert metrics.evidence_source_count == len(context.evidence_catalog)
    assert "signal_review_2" in {
        source.evidence_id for source in report.sources_used
    }
