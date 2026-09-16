import asyncio
import json

import httpx
import pytest

from app.schemas.web_research import (
    TavilyExtractRequest,
    TavilySearchRequest,
)
from app.search.tavily_provider import (
    TavilyConfigurationError,
    TavilyProvider,
    TavilyQuotaExceededError,
)


def json_response(payload: dict, status_code: int = 200) -> httpx.Response:
    return httpx.Response(
        status_code,
        content=json.dumps(payload),
        headers={"Content-Type": "application/json"},
    )


def test_search_returns_normalized_results_and_usage():
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/search"
        assert request.headers["Authorization"] == "Bearer test-key"

        payload = json.loads(request.content)
        assert payload["search_depth"] == "basic"
        assert payload["max_results"] == 5
        assert payload["include_answer"] is False
        assert payload["include_raw_content"] is False
        assert payload["include_usage"] is True
        assert payload["safe_search"] is True

        return json_response(
            {
                "query": payload["query"],
                "results": [
                    {
                        "title": "India beverage market",
                        "url": "https://example.com/market",
                        "content": "Market evidence snippet.",
                        "score": 0.91,
                        "published_date": "2026-09-01",
                    }
                ],
                "usage": {"credits": 1},
                "response_time": 0.4,
                "request_id": "search-request-1",
            }
        )

    provider = TavilyProvider(
        api_key="test-key",
        transport=httpx.MockTransport(handler),
    )

    result = asyncio.run(
        provider.search(
            TavilySearchRequest(
                query="Coca-Cola competitors India beverage market",
            )
        )
    )

    assert result.results[0].title == "India beverage market"
    assert str(result.results[0].url) == "https://example.com/market"
    assert result.usage.credits_used == 1
    assert result.request_id == "search-request-1"


def test_extract_returns_successes_failures_and_usage():
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/extract"
        payload = json.loads(request.content)
        assert len(payload["urls"]) == 2
        assert payload["query"] == "beverage entry barriers"
        assert payload["chunks_per_source"] == 3

        return json_response(
            {
                "results": [
                    {
                        "url": "https://example.com/market",
                        "raw_content": "Clean extracted market evidence.",
                    }
                ],
                "failed_results": [
                    {
                        "url": "https://example.com/blocked",
                        "error": "Page blocked extraction",
                    }
                ],
                "usage": {"credits": 1},
                "response_time": 0.8,
                "request_id": "extract-request-1",
            }
        )

    provider = TavilyProvider(
        api_key="test-key",
        transport=httpx.MockTransport(handler),
    )

    result = asyncio.run(
        provider.extract(
            TavilyExtractRequest(
                urls=[
                    "https://example.com/market",
                    "https://example.com/blocked",
                ],
                query="beverage entry barriers",
            )
        )
    )

    assert len(result.documents) == 1
    assert len(result.failed_documents) == 1
    assert result.usage.credits_used == 1


def test_missing_api_key_is_rejected(monkeypatch):
    monkeypatch.setattr(
        "app.search.tavily_provider.settings.tavily_api_key",
        "",
    )

    with pytest.raises(
        TavilyConfigurationError,
        match="TAVILY_API_KEY",
    ):
        TavilyProvider(api_key="")


def test_credit_limit_has_specific_error():
    async def handler(_: httpx.Request) -> httpx.Response:
        return json_response(
            {"detail": {"error": "Usage limit reached"}},
            status_code=432,
        )

    provider = TavilyProvider(
        api_key="test-key",
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(TavilyQuotaExceededError):
        asyncio.run(
            provider.search(
                TavilySearchRequest(query="test company market research")
            )
        )
