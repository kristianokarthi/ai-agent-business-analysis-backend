import asyncio

from app.collectors.public_signal_collector import (
    PublicSignalCollector,
    build_public_signal_queries,
    classify_signal_source,
    is_usable_signal_content,
    redact_contact_details,
    select_public_signal_results,
)
from app.schemas.public_signal_evidence import PublicSignalCollectionInput
from app.schemas.web_research import (
    ExtractedDocument,
    FailedExtraction,
    SearchResult,
    TavilyExtractResponse,
    TavilySearchResponse,
    TavilyUsage,
)


def collection_input() -> PublicSignalCollectionInput:
    return PublicSignalCollectionInput(
        company_name="The Coca-Cola Company",
        purpose="stock_research",
        follow_up_answers={"geographic_market": "India"},
    )


def search_response(
    query: str,
    results: list[dict],
) -> TavilySearchResponse:
    return TavilySearchResponse(
        query=query,
        results=[SearchResult.model_validate(item) for item in results],
        usage=TavilyUsage(
            operation="search",
            credits_used=1,
            response_time_seconds=0.2,
        ),
        request_id=f"search-{query}",
    )


class FakePublicSignalProvider:
    def __init__(
        self,
        searches: list[TavilySearchResponse],
        extraction: TavilyExtractResponse,
    ) -> None:
        self.searches = searches
        self.extraction = extraction
        self.search_requests = []
        self.extract_requests = []

    async def search(self, request):
        self.search_requests.append(request)
        return self.searches[len(self.search_requests) - 1]

    async def extract(self, request):
        self.extract_requests.append(request)
        return self.extraction


def test_stock_queries_are_company_and_reputation_specific():
    queries = build_public_signal_queries(collection_input())

    assert len(queries) == 3
    assert all("The Coca-Cola Company" in query for query, _ in queries)
    assert all("India" in query for query, _ in queries)
    assert "customer reviews" in queries[0][0]
    assert "consumer survey" in queries[1][0]
    assert queries[2][1] == "news"


def test_signal_selection_deduplicates_and_preserves_search_coverage():
    responses = [
        search_response(
            "reviews",
            [
                {
                    "title": "Coca-Cola reviews",
                    "url": "https://example.com/reviews/",
                    "content": "Customer feedback",
                    "relevance_score": 0.8,
                }
            ],
        ),
        search_response(
            "reputation",
            [
                {
                    "title": "Updated Coca-Cola reviews",
                    "url": "https://example.com/reviews",
                    "content": "More customer feedback",
                    "relevance_score": 0.9,
                },
                {
                    "title": "Coca-Cola reputation",
                    "url": "https://example.com/reputation",
                    "content": "Reputation report",
                    "relevance_score": 0.85,
                },
            ],
        ),
    ]

    selected = select_public_signal_results(
        responses,
        "The Coca-Cola Company",
    )

    assert len(selected) == 2
    assert selected[0].title == "Updated Coca-Cola reviews"
    assert selected[1].title == "Coca-Cola reputation"


def test_signal_selection_rejects_irrelevant_and_weak_sources():
    responses = [
        search_response(
            "signals",
            [
                {
                    "title": "Contact Us",
                    "url": "https://www.coca-cola.com/in/en/about-us/contact-us",
                    "content": "Consumer related queries and contact details.",
                    "relevance_score": 0.99,
                },
                {
                    "title": "Coca-Cola Marketing Report Example",
                    "url": "https://ivypanda.com/essays/coca-cola-marketing",
                    "content": "A free essay about consumer perception.",
                    "relevance_score": 0.98,
                },
                {
                    "title": "Hexagon Nutrition IPO risks",
                    "url": "https://www.livemint.com/market/ipo/hexagon",
                    "content": "An unrelated company faces customer risk.",
                    "relevance_score": 0.97,
                },
                {
                    "title": "Coca-Cola customer satisfaction survey",
                    "url": "https://research.example.org/coca-cola-survey",
                    "content": "Consumers describe Coca-Cola product quality.",
                    "relevance_score": 0.8,
                },
            ],
        )
    ]

    selected = select_public_signal_results(
        responses,
        "The Coca-Cola Company",
    )

    assert [item.title for item in selected] == [
        "Coca-Cola customer satisfaction survey"
    ]


def test_extracted_signal_quality_gate_rejects_empty_or_unrelated_content():
    assert not is_usable_signal_content(
        company_name="The Coca-Cola Company",
        title="Consumer preference sample",
        url="https://example.com/sample",
        content="``` [...] ```",
    )
    assert not is_usable_signal_content(
        company_name="The Coca-Cola Company",
        title="Hexagon Nutrition IPO risks",
        url="https://example.com/hexagon",
        content=(
            "Customers should review this unrelated company's litigation "
            "and product risks before investing. " * 3
        ),
    )
    assert is_usable_signal_content(
        company_name="The Coca-Cola Company",
        title="Coca-Cola customer satisfaction survey",
        url="https://example.com/coca-cola-survey",
        content=(
            "A consumer survey of Coca-Cola customers reported strong brand "
            "recognition alongside recurring feedback about sweetness and "
            "interest in lower-sugar products in the Indian market."
        ),
    )


def test_source_classification_and_contact_redaction():
    assert classify_signal_source(
        "https://www.trustpilot.com/review/example.com",
        "Customer reviews",
    ).value == "customer_review"
    assert classify_signal_source(
        "https://www.reuters.com/business/example",
        "Company reputation report",
    ).value == "news"

    cleaned = redact_contact_details(
        "Email user@example.com, call +91 98765 43210, or ask @reviewer."
    )
    assert "user@example.com" not in cleaned
    assert "98765" not in cleaned
    assert "@reviewer" not in cleaned


def test_collector_returns_agent_4_signals_and_usage():
    searches = [
        search_response(
            "reviews",
            [
                {
                    "title": "Coca-Cola customer reviews",
                    "url": "https://www.trustpilot.com/review/coca-cola.com",
                    "content": "Customer review summary",
                    "relevance_score": 0.95,
                }
            ],
        ),
        search_response(
            "complaints",
            [
                {
                    "title": "Coca-Cola discussion",
                    "url": "https://www.reddit.com/r/example/comments/cola",
                    "content": "Public forum discussion",
                    "relevance_score": 0.9,
                }
            ],
        ),
        search_response(
            "news",
            [
                {
                    "title": "Coca-Cola reputation news",
                    "url": "https://www.reuters.com/business/coca-cola",
                    "content": "Reputation news summary",
                    "relevance_score": 0.92,
                    "published_date": "2026-09-01",
                }
            ],
        ),
    ]
    extraction = TavilyExtractResponse(
        documents=[
            ExtractedDocument(
                url="https://www.trustpilot.com/review/coca-cola.com",
                content=(
                    "Coca-Cola customers praise widespread product availability "
                    "and consistent taste, but several customer reviews mention "
                    "excessive sweetness and ask for more low-sugar choices."
                ),
            ),
            ExtractedDocument(
                url="https://www.reddit.com/r/example/comments/cola",
                content=(
                    "A public Coca-Cola discussion contains mixed product "
                    "feedback. Consumers appreciate availability while some "
                    "customers complain about sweetness and packaging waste."
                ),
            ),
        ],
        failed_documents=[
            FailedExtraction(
                url="https://www.reuters.com/business/coca-cola",
                error="Page blocked extraction",
            )
        ],
        usage=TavilyUsage(
            operation="extract",
            credits_used=0,
            response_time_seconds=0.4,
        ),
        request_id="extract-signals-1",
    )
    provider = FakePublicSignalProvider(searches, extraction)

    result = asyncio.run(
        PublicSignalCollector(provider=provider).collect(
            collection_input()
        )
    )

    assert result.status.value == "completed"
    assert len(result.signal_documents) == 2
    assert result.signal_documents[0].source_type.value == "customer_review"
    assert result.signal_documents[1].source_type.value == "forum"
    assert result.usage.search_requests == 3
    assert result.usage.extraction_requests == 1
    assert result.usage.selected_urls == 3
    assert result.usage.extracted_documents == 2
    assert result.usage.credits_used == 3
    assert len(result.failed_sources) == 1


def test_collector_skips_extraction_when_search_is_empty():
    searches = [
        search_response("one", []),
        search_response("two", []),
        search_response("three", []),
    ]
    unused_extraction = TavilyExtractResponse(
        documents=[],
        failed_documents=[],
        usage=TavilyUsage(
            operation="extract",
            credits_used=0,
            response_time_seconds=0,
        ),
    )
    provider = FakePublicSignalProvider(searches, unused_extraction)

    result = asyncio.run(
        PublicSignalCollector(provider=provider).collect(
            collection_input()
        )
    )

    assert result.status.value == "insufficient_sources"
    assert result.signal_documents == []
    assert result.usage.extraction_requests == 0
    assert provider.extract_requests == []
