from fastapi.testclient import TestClient

from app.agents.report_strategist import finalize_report
from app.main import app
from app.rag.report_chunker import chunk_strategic_report
from app.reports.context_builder import build_report_context
from app.schemas.rag import ReportChunkSection
from app.schemas.report_context import ReportContextRequest
from app.schemas.report_strategist import ReportStrategistDraft
from app.tests.test_report_context_builder import valid_request_data
from app.tests.test_report_strategist_agent import valid_draft_data


def strategic_report():
    context = build_report_context(
        ReportContextRequest.model_validate(valid_request_data())
    )
    draft = ReportStrategistDraft.model_validate(valid_draft_data())
    return finalize_report(draft, context)


def test_chunker_keeps_each_report_insight_intact():
    report = strategic_report()

    result = chunk_strategic_report(report)

    executive = result.chunks[0]
    assert executive.chunk_id == "executive_summary_01"
    assert executive.title == report.executive_summary.title
    assert executive.content == report.executive_summary.analysis
    assert executive.evidence_ids == ["source_1"]
    assert executive.sources[0].evidence_id == "source_1"


def test_chunker_creates_expected_section_counts():
    report = strategic_report()

    result = chunk_strategic_report(report)

    assert result.company_name == "The Coca-Cola Company"
    assert result.total_chunks == len(result.chunks)
    assert result.section_counts[
        ReportChunkSection.IMPORTANT_FINDING
    ] == 2
    assert result.section_counts[ReportChunkSection.RISK] == 1
    assert result.section_counts[
        ReportChunkSection.BALANCED_CONCLUSION
    ] == 1


def test_non_factual_chunks_do_not_invent_sources():
    report = strategic_report()

    result = chunk_strategic_report(report)
    missing_chunks = [
        chunk
        for chunk in result.chunks
        if chunk.section == ReportChunkSection.MISSING_INFORMATION
    ]

    assert missing_chunks
    assert all(not chunk.evidence_ids for chunk in missing_chunks)
    assert all(not chunk.sources for chunk in missing_chunks)


def test_all_chunk_ids_are_unique():
    result = chunk_strategic_report(strategic_report())
    chunk_ids = [chunk.chunk_id for chunk in result.chunks]

    assert len(chunk_ids) == len(set(chunk_ids))


def test_chunk_preview_endpoint_returns_generated_chunks():
    client = TestClient(app)
    report = strategic_report()

    response = client.post(
        "/api/rag/chunks/preview",
        json={"report": report.model_dump(mode="json")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["company_name"] == "The Coca-Cola Company"
    assert payload["total_chunks"] == len(payload["chunks"])
