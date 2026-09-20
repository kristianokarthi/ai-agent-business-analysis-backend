from collections import Counter

from app.schemas.rag import (
    ReportChunk,
    ReportChunkPreviewResponse,
    ReportChunkSection,
)
from app.schemas.report_strategist import (
    ReportInsight,
    StrategicReport,
)


def _chunk_id(section: ReportChunkSection, position: int) -> str:
    return f"{section.value}_{position:02d}"


def _insight_chunk(
    insight: ReportInsight,
    section: ReportChunkSection,
    position: int,
    source_map: dict,
) -> ReportChunk:
    sources = [
        source_map[evidence_id]
        for evidence_id in insight.evidence_ids
        if evidence_id in source_map
    ]
    return ReportChunk(
        chunk_id=_chunk_id(section, position),
        section=section,
        title=insight.title,
        content=insight.analysis,
        evidence_ids=insight.evidence_ids,
        sources=sources,
    )


def _text_chunk(
    content: str,
    section: ReportChunkSection,
    position: int,
    title: str,
) -> ReportChunk:
    return ReportChunk(
        chunk_id=_chunk_id(section, position),
        section=section,
        title=title,
        content=content,
    )


def chunk_strategic_report(
    report: StrategicReport,
) -> ReportChunkPreviewResponse:
    """Turn one structured Agent 5 report into semantic RAG chunks.

    Each report insight remains intact. No embedding, database, LLM, or
    token-based splitting is performed in this learning stage.
    """
    source_map = {
        source.evidence_id: source
        for source in report.sources_used
    }
    chunks: list[ReportChunk] = []

    chunks.append(
        _insight_chunk(
            report.executive_summary,
            ReportChunkSection.EXECUTIVE_SUMMARY,
            1,
            source_map,
        )
    )

    insight_groups = [
        (
            ReportChunkSection.IMPORTANT_FINDING,
            report.important_findings,
        ),
        (ReportChunkSection.OPPORTUNITY, report.opportunities),
        (ReportChunkSection.RISK, report.risks),
        (
            ReportChunkSection.KEY_CONSIDERATION,
            report.purpose_specific_analysis.key_considerations,
        ),
        (
            ReportChunkSection.FAVORABLE_CASE,
            report.purpose_specific_analysis.favorable_case,
        ),
        (
            ReportChunkSection.CAUTION_CASE,
            report.purpose_specific_analysis.caution_case,
        ),
        (
            ReportChunkSection.DECISION_FACTOR,
            report.purpose_specific_analysis.decision_factors,
        ),
    ]
    for section, insights in insight_groups:
        chunks.extend(
            _insight_chunk(
                insight,
                section,
                position,
                source_map,
            )
            for position, insight in enumerate(insights, start=1)
        )

    chunks.append(
        _insight_chunk(
            report.balanced_conclusion,
            ReportChunkSection.BALANCED_CONCLUSION,
            1,
            source_map,
        )
    )

    text_groups = [
        (
            ReportChunkSection.INVESTIGATION_QUESTION,
            "Question for further investigation",
            report.questions_for_further_investigation,
        ),
        (
            ReportChunkSection.LIMITATION,
            "Evidence limitation",
            report.confidence_assessment.limitations,
        ),
        (
            ReportChunkSection.MISSING_INFORMATION,
            "Missing information",
            report.missing_information,
        ),
        (
            ReportChunkSection.QUALITY_WARNING,
            "Quality warning",
            report.quality_warnings,
        ),
    ]
    for section, title, items in text_groups:
        chunks.extend(
            _text_chunk(
                item,
                section,
                position,
                f"{title} {position}",
            )
            for position, item in enumerate(items, start=1)
        )

    section_counts = Counter(chunk.section for chunk in chunks)
    return ReportChunkPreviewResponse(
        company_name=report.company_name,
        purpose=report.purpose,
        total_chunks=len(chunks),
        section_counts=dict(section_counts),
        chunks=chunks,
    )
