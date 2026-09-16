import asyncio

from app.collectors.market_collector import (
    MarketEvidenceCollector,
    build_market_search_queries,
    select_market_results,
)
from app.schemas.market_evidence import MarketEvidenceCollectionInput
from app.schemas.web_research import (
    ExtractedDocument,
    FailedExtraction,
    SearchResult,
    TavilyExtractResponse,
    TavilySearchResponse,
    TavilyUsage,
)
from app.tests.test_market_competitor_schema import (
    VALID_BUSINESS_FUNDAMENTALS,
    VALID_COMPANY_EVIDENCE,
)


def collection_input() -> MarketEvidenceCollectionInput:
    return MarketEvidenceCollectionInput.model_validate(
        {
            "purpose": "start_similar_business",
            "follow_up_answers": {"target_location": "Chennai"},
            "company_evidence": VALID_COMPANY_EVIDENCE,
            "business_fundamentals": VALID_BUSINESS_FUNDAMENTALS,
        }
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


class FakeMarketProvider:
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


def test_query_builder_uses_company_industry_purpose_and_location():
    queries = build_market_search_queries(collection_input())

    assert len(queries) == 3
    assert all("The Coca-Cola Company" in query for query in queries)
    assert all("Beverages" in query for query in queries)
    assert any("new entrant" in query for query in queries)
    assert all("Chennai" in query for query in queries)


def test_stock_queries_are_company_specific_and_investment_relevant():
    request = MarketEvidenceCollectionInput.model_validate(
        {
            **collection_input().model_dump(),
            "purpose": "stock_research",
        }
    )

    queries = build_market_search_queries(request)

    assert all("The Coca-Cola Company" in query for query in queries)
    assert "market share" in queries[0]
    assert "industry outlook" in queries[1]
    assert "annual report" in queries[2]


def test_result_selection_deduplicates_urls_and_keeps_best_score():
    responses = [
        search_response(
            "one",
            [
                {
                    "title": "First title",
                    "url": "https://example.com/market/",
                    "content": "First snippet",
                    "relevance_score": 0.7,
                }
            ],
        ),
        search_response(
            "two",
            [
                {
                    "title": "Better title",
                    "url": "https://example.com/market",
                    "content": "Better snippet",
                    "relevance_score": 0.9,
                }
            ],
        ),
    ]

    selected = select_market_results(responses)

    assert len(selected) == 1
    assert selected[0].title == "Better title"


def test_collector_returns_agent_3_documents_and_aggregated_usage():
    searches = [
        search_response(
            "competitors",
            [
                {
                    "title": "India beverage market report",
                    "url": "https://www.kenresearch.com/india-beverages",
                    "content": "Market snippet",
                    "relevance_score": 0.95,
                },
                {
                    "title": "Coca-Cola Company",
                    "url": "https://www.coca-colacompany.com/strategy",
                    "content": "Company snippet",
                    "relevance_score": 0.8,
                },
            ],
        ),
        search_response(
            "trends",
            [
                {
                    "title": "Updated market report",
                    "url": "https://www.kenresearch.com/india-beverages/",
                    "content": "Updated snippet",
                    "relevance_score": 0.98,
                },
                {
                    "title": "Soft drink competition news",
                    "url": "https://timesofindia.indiatimes.com/soft-drinks",
                    "content": "News snippet",
                    "relevance_score": 0.9,
                },
            ],
        ),
        search_response("barriers", []),
    ]
    extraction = TavilyExtractResponse(
        documents=[
            ExtractedDocument(
                url="https://www.kenresearch.com/india-beverages",
                content="Detailed market evidence.",
            ),
            ExtractedDocument(
                url="https://timesofindia.indiatimes.com/soft-drinks",
                content="Detailed competition news.",
            ),
        ],
        failed_documents=[
            FailedExtraction(
                url="https://www.coca-colacompany.com/strategy",
                error="Page could not be extracted",
            )
        ],
        usage=TavilyUsage(
            operation="extract",
            credits_used=0,
            response_time_seconds=0.4,
        ),
        request_id="extract-1",
    )
    provider = FakeMarketProvider(searches, extraction)

    result = asyncio.run(
        MarketEvidenceCollector(provider=provider).collect(
            collection_input()
        )
    )

    assert result.status.value == "completed"
    assert len(result.market_documents) == 2
    assert result.market_documents[0].source_id == "market_source_1"
    assert result.market_documents[0].title == "Updated market report"
    assert result.market_documents[0].source_type.value == "industry_report"
    assert result.market_documents[1].source_type.value == "news"
    assert result.usage.search_requests == 3
    assert result.usage.extraction_requests == 1
    assert result.usage.searched_results == 4
    assert result.usage.selected_urls == 3
    assert result.usage.credits_used == 3
    assert len(result.failed_sources) == 1
    assert len(provider.extract_requests[0].urls) == 3


def test_collector_does_not_extract_when_search_has_no_results():
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
    provider = FakeMarketProvider(searches, unused_extraction)

    result = asyncio.run(
        MarketEvidenceCollector(provider=provider).collect(
            collection_input()
        )
    )

    assert result.status.value == "insufficient_sources"
    assert result.market_documents == []
    assert result.usage.extraction_requests == 0
    assert provider.extract_requests == []
