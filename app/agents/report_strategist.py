import re
from collections.abc import Iterable

from app.llm.openrouter_provider import OpenRouterProvider
from app.llm.types import StructuredLLMResult
from app.prompts.report_strategist import build_report_strategist_prompt
from app.schemas.report_context import ReportContext
from app.schemas.report_strategist import (
    ReportCitationSource,
    ReportContextMetrics,
    ReportInsight,
    ReportStrategistDraft,
    StrategicReport,
)
from app.schemas.research import ResearchPurpose


class InvalidReportEvidenceError(ValueError):
    """Raised when Agent 5 changes identity or cites absent evidence."""


class UnsafeStockRecommendationError(ValueError):
    """Raised when Agent 5 returns a direct trading instruction."""


def _report_insights(report: ReportStrategistDraft) -> Iterable[ReportInsight]:
    yield report.executive_summary
    yield from report.important_findings
    yield from report.opportunities
    yield from report.risks
    yield from report.purpose_specific_analysis.key_considerations
    yield from report.purpose_specific_analysis.favorable_case
    yield from report.purpose_specific_analysis.caution_case
    yield from report.purpose_specific_analysis.decision_factors
    yield report.balanced_conclusion


def _report_text(report: ReportStrategistDraft) -> str:
    values = [
        report.title,
        *(insight.title for insight in _report_insights(report)),
        *(insight.analysis for insight in _report_insights(report)),
        *report.questions_for_further_investigation,
        report.confidence_assessment.rationale,
        *report.confidence_assessment.limitations,
        *report.missing_information,
    ]
    return " ".join(values)


def validate_report_draft(
    report: ReportStrategistDraft,
    context: ReportContext,
) -> None:
    report_company = report.company_name.strip().casefold()
    context_company = context.company_name.strip().casefold()
    if report_company != context_company:
        raise InvalidReportEvidenceError(
            "Agent 5 returned a different company name."
        )
    if report.purpose != context.purpose:
        raise InvalidReportEvidenceError(
            "Agent 5 returned a different research purpose."
        )

    available_ids = {
        source.evidence_id for source in context.evidence_catalog
    }
    for insight in _report_insights(report):
        unknown_ids = set(insight.evidence_ids) - available_ids
        if unknown_ids:
            raise InvalidReportEvidenceError(
                f"Report item '{insight.title}' cited unknown evidence: "
                + ", ".join(sorted(unknown_ids))
            )

    if context.purpose == ResearchPurpose.STOCK_RESEARCH:
        text = _report_text(report)
        prohibited = re.compile(
            r"\b(?:recommend(?:ation)?\s+(?:is\s+)?(?:to\s+)?)?"
            r"(?:buy|sell|hold)\s+(?:the\s+)?(?:stock|shares?)\b",
            re.IGNORECASE,
        )
        if prohibited.search(text):
            raise UnsafeStockRecommendationError(
                "Agent 5 returned a direct stock trading instruction."
            )


def finalize_report(
    draft: ReportStrategistDraft,
    context: ReportContext,
) -> StrategicReport:
    missing_information = list(
        dict.fromkeys(
            [*context.missing_information, *draft.missing_information]
        )
    )
    limitations = list(
        dict.fromkeys(
            [
                *draft.confidence_assessment.limitations,
                *context.missing_information,
            ]
        )
    )[:30]

    cited_ids: list[str] = []
    for insight in _report_insights(draft):
        for evidence_id in insight.evidence_ids:
            if evidence_id not in cited_ids:
                cited_ids.append(evidence_id)

    catalog = {
        source.evidence_id: source for source in context.evidence_catalog
    }
    sources = [
        ReportCitationSource(
            evidence_id=evidence_id,
            title=catalog[evidence_id].title,
            url=catalog[evidence_id].url,
            source_type=catalog[evidence_id].source_type,
        )
        for evidence_id in cited_ids
    ]

    draft_data = draft.model_dump()
    draft_data["missing_information"] = missing_information[:30]
    draft_data["confidence_assessment"]["limitations"] = limitations

    provisional_text = _report_text(
        ReportStrategistDraft.model_validate(draft_data)
    )
    word_count = len(provisional_text.split())
    quality_warnings: list[str] = []
    if word_count < 1_000:
        quality_warnings.append(
            "The generated report is shorter than the 1,200-word target."
        )
    elif word_count > 1_900:
        quality_warnings.append(
            "The generated report exceeds the preferred overview length."
        )
    if context.truncated_for_budget:
        quality_warnings.append(
            "Optional context findings were trimmed to respect the token budget."
        )

    return StrategicReport(
        **draft_data,
        sources_used=sources,
        word_count=word_count,
        quality_warnings=quality_warnings,
    )


def context_metrics(context: ReportContext) -> ReportContextMetrics:
    return ReportContextMetrics(
        estimated_tokens=context.estimated_tokens,
        token_budget=context.token_budget,
        truncated_for_budget=context.truncated_for_budget,
        evidence_source_count=len(context.evidence_catalog),
    )


class ReportStrategistAgent:
    def __init__(self, provider: OpenRouterProvider | None = None) -> None:
        self.provider = provider or OpenRouterProvider()

    async def run(
        self,
        context: ReportContext,
    ) -> StructuredLLMResult[StrategicReport]:
        allowed_evidence_ids = [
            source.evidence_id for source in context.evidence_catalog
        ]
        response = await self.provider.generate_structured(
            agent_name="report_strategist",
            system_prompt=build_report_strategist_prompt(context.purpose),
            user_prompt=(
                "Create the final evidence-backed report from this compressed "
                "context. Every evidence_ids array may contain only values "
                "from the allowed citation list below. Do not cite finding_id "
                "values.\n\n"
                "Allowed citation evidence IDs:\n"
                f"{allowed_evidence_ids}\n\n"
                "Compressed report context:\n"
                f"{context.model_dump_json(indent=2)}"
            ),
            response_model=ReportStrategistDraft,
            max_tokens=6_000,
            temperature=0,
        )

        validate_report_draft(response.data, context)
        report = finalize_report(response.data, context)
        return StructuredLLMResult(data=report, usage=response.usage)
