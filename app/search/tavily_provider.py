import logging
from typing import Any

import httpx
from pydantic import ValidationError

from app.core.config import settings
from app.schemas.web_research import (
    ExtractedDocument,
    FailedExtraction,
    SearchResult,
    TavilyExtractRequest,
    TavilyExtractResponse,
    TavilySearchRequest,
    TavilySearchResponse,
    TavilyUsage,
)


TAVILY_API_BASE = "https://api.tavily.com"
REQUEST_TIMEOUT_SECONDS = 30.0

logger = logging.getLogger("uvicorn.error")


class TavilyProviderError(Exception):
    pass


class TavilyConfigurationError(TavilyProviderError):
    pass


class TavilyRequestError(TavilyProviderError):
    pass


class TavilyRateLimitError(TavilyProviderError):
    pass


class TavilyQuotaExceededError(TavilyProviderError):
    pass


class TavilyServiceError(TavilyProviderError):
    pass


class TavilyProvider:
    def __init__(
        self,
        api_key: str | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.api_key = api_key or settings.tavily_api_key
        self.transport = transport

        if not self.api_key:
            raise TavilyConfigurationError(
                "TAVILY_API_KEY is missing from environment variables."
            )

    async def _post(
        self,
        path: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        try:
            async with httpx.AsyncClient(
                base_url=TAVILY_API_BASE,
                timeout=REQUEST_TIMEOUT_SECONDS,
                transport=self.transport,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
            ) as client:
                response = await client.post(path, json=payload)
        except httpx.TimeoutException as error:
            raise TavilyServiceError(
                "Tavily did not respond before the request timed out."
            ) from error
        except httpx.HTTPError as error:
            raise TavilyServiceError(
                "Tavily could not be reached."
            ) from error

        if response.status_code == 401:
            raise TavilyConfigurationError(
                "The Tavily API key is missing or invalid."
            )
        if response.status_code == 429:
            raise TavilyRateLimitError(
                "Tavily's request rate limit has been reached."
            )
        if response.status_code in {432, 433}:
            raise TavilyQuotaExceededError(
                "The Tavily usage limit has been reached."
            )
        if response.status_code in {400, 422}:
            raise TavilyRequestError(
                "Tavily rejected the search request."
            )
        if 400 <= response.status_code < 500:
            raise TavilyRequestError(
                "Tavily rejected the search request."
            )
        if response.status_code >= 500:
            raise TavilyServiceError(
                "Tavily is temporarily unavailable."
            )

        try:
            response.raise_for_status()
            body = response.json()
        except (httpx.HTTPError, ValueError) as error:
            raise TavilyServiceError(
                "Tavily returned an unreadable response."
            ) from error

        if not isinstance(body, dict):
            raise TavilyServiceError(
                "Tavily returned an unexpected response."
            )

        return body

    async def search(
        self,
        request: TavilySearchRequest,
    ) -> TavilySearchResponse:
        payload: dict[str, Any] = {
            "query": request.query,
            "topic": request.topic.value,
            "search_depth": request.search_depth.value,
            "chunks_per_source": 2,
            "max_results": request.max_results,
            "include_answer": False,
            "include_raw_content": False,
            "include_images": False,
            "include_published_date": True,
            "include_usage": True,
            "safe_search": True,
            "auto_parameters": False,
        }

        if request.country and request.topic.value == "general":
            payload["country"] = request.country.lower()
        if request.time_range:
            payload["time_range"] = request.time_range

        body = await self._post("/search", payload)

        try:
            results = [
                SearchResult(
                    title=item.get("title") or "Untitled source",
                    url=item["url"],
                    content=item.get("content") or "",
                    relevance_score=item.get("score") or 0,
                    published_date=item.get("published_date"),
                )
                for item in body.get("results", [])
            ]
            usage = TavilyUsage(
                operation="search",
                credits_used=body.get("usage", {}).get("credits", 0),
                response_time_seconds=float(body.get("response_time") or 0),
            )
            result = TavilySearchResponse(
                query=body.get("query") or request.query,
                results=results,
                usage=usage,
                request_id=body.get("request_id"),
            )
        except (KeyError, TypeError, ValueError, ValidationError) as error:
            raise TavilyServiceError(
                "Tavily returned an invalid search response."
            ) from error

        logger.info(
            "Search usage | provider=tavily | operation=search | "
            "credits=%s | results=%d | request_id=%s",
            result.usage.credits_used,
            len(result.results),
            result.request_id or "unknown",
        )

        return result

    async def extract(
        self,
        request: TavilyExtractRequest,
    ) -> TavilyExtractResponse:
        body = await self._post(
            "/extract",
            {
                "urls": [str(url) for url in request.urls],
                "query": request.query,
                "chunks_per_source": request.chunks_per_source,
                "extract_depth": request.extract_depth.value,
                "include_images": False,
                "format": "markdown",
                "include_usage": True,
            },
        )

        try:
            documents = [
                ExtractedDocument(
                    url=item["url"],
                    content=item.get("raw_content") or "",
                )
                for item in body.get("results", [])
                if item.get("raw_content")
            ]
            failed_documents = [
                FailedExtraction(
                    url=item["url"],
                    error=item.get("error") or "Extraction failed",
                )
                for item in body.get("failed_results", [])
            ]
            usage = TavilyUsage(
                operation="extract",
                credits_used=body.get("usage", {}).get("credits", 0),
                response_time_seconds=float(body.get("response_time") or 0),
            )
            result = TavilyExtractResponse(
                documents=documents,
                failed_documents=failed_documents,
                usage=usage,
                request_id=body.get("request_id"),
            )
        except (KeyError, TypeError, ValueError, ValidationError) as error:
            raise TavilyServiceError(
                "Tavily returned an invalid extraction response."
            ) from error

        logger.info(
            "Search usage | provider=tavily | operation=extract | "
            "credits=%s | successful=%d | failed=%d | request_id=%s",
            result.usage.credits_used,
            len(result.documents),
            len(result.failed_documents),
            result.request_id or "unknown",
        )

        return result
