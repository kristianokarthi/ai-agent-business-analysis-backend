import asyncio
import re
from collections.abc import Iterable
from typing import Protocol
from urllib.parse import urlsplit, urlunsplit

from app.schemas.market_competitor import (
    MarketSourceDocument,
    MarketSourceType,
)
from app.schemas.market_evidence import (
    FailedMarketSource,
    MarketCollectionStatus,
    MarketCollectorUsage,
    MarketEvidenceCollectionInput,
    MarketEvidenceCollectionOutput,
)
from app.schemas.web_research import (
    SearchResult,
    TavilyExtractRequest,
    TavilyExtractResponse,
    TavilySearchRequest,
    TavilySearchResponse,
)
from app.search.tavily_provider import TavilyProvider


MAX_MARKET_SEARCHES = 3
MAX_MARKET_DOCUMENTS = 5
MAX_DOCUMENT_CHARACTERS = 12_000


class MarketResearchProvider(Protocol):
    async def search(
        self,
        request: TavilySearchRequest,
    ) -> TavilySearchResponse: ...

    async def extract(
        self,
        request: TavilyExtractRequest,
    ) -> TavilyExtractResponse: ...


def _answer_for(
    answers: dict[str, str],
    keys: Iterable[str],
) -> str | None:
    normalized_answers = {
        key.strip().casefold(): value.strip()
        for key, value in answers.items()
        if value.strip()
    }
    for key in keys:
        value = normalized_answers.get(key.casefold())
        if value:
            return value
    return None


def _compact_query(*parts: str) -> str:
    query = " ".join(" ".join(parts).split())
    return query[:300].rstrip()


def build_market_search_queries(
    request: MarketEvidenceCollectionInput,
) -> list[str]:
    identity = request.company_evidence.company_identity
    company_name = identity.name
    industry = identity.industry or "business sector"
    location = _answer_for(
        request.follow_up_answers,
        (
            "geographic_market",
            "target_location",
            "target_market",
            "location",
            "region",
            "country",
            "market",
        ),
    ) or "global"

    purpose_queries = {
        "general_research": (
            "main competitors and market position",
            "market share and competitive landscape",
            "industry trends competition and risks",
        ),
        "start_similar_business": (
            "main competitors and alternatives",
            "new entrant opportunities and market gaps",
            "entry barriers pricing distribution and regulation",
        ),
        "partner_or_supplier": (
            "competitors suppliers and distribution partners",
            "supply chain channels and market access",
            "partnership ecosystem and competitive conflicts",
        ),
        "stock_research": (
            "main competitors and market share",
            "competitive position and industry outlook",
            "competition risks and annual report",
        ),
    }[request.purpose.value]

    return [
        _compact_query(
            company_name,
            industry,
            focus,
            location,
        )
        for focus in purpose_queries
    ][:MAX_MARKET_SEARCHES]


def _canonical_url(url: str) -> str:
    parsed = urlsplit(url)
    normalized_path = parsed.path.rstrip("/") or "/"
    return urlunsplit(
        (
            parsed.scheme.casefold(),
            parsed.netloc.casefold(),
            normalized_path,
            parsed.query,
            "",
        )
    )


def select_market_results(
    responses: list[TavilySearchResponse],
    company_name: str | None = None,
) -> list[SearchResult]:
    unique_results: dict[str, SearchResult] = {}

    for response in responses:
        for result in response.results:
            key = _canonical_url(str(result.url))
            existing = unique_results.get(key)
            if (
                existing is None
                or result.relevance_score > existing.relevance_score
            ):
                unique_results[key] = result

    ignored_terms = {"the", "company", "corporation", "inc", "limited"}
    company_terms = {
        term
        for term in re.findall(
            r"[a-z0-9]+",
            (company_name or "").casefold(),
        )
        if len(term) > 2 and term not in ignored_terms
    }

    def result_score(result: SearchResult) -> float:
        searchable = f"{result.title} {result.content}".casefold()
        company_bonus = (
            0.2
            if any(term in searchable for term in company_terms)
            else 0
        )
        return result.relevance_score + company_bonus

    selected: list[SearchResult] = []
    selected_urls: set[str] = set()

    # Preserve coverage across every search intent before filling the
    # remaining positions by global relevance.
    for response in responses:
        candidates = sorted(
            response.results,
            key=result_score,
            reverse=True,
        )
        for candidate in candidates:
            key = _canonical_url(str(candidate.url))
            if key not in selected_urls:
                selected.append(unique_results[key])
                selected_urls.add(key)
                break

    remaining = sorted(
        unique_results.values(),
        key=result_score,
        reverse=True,
    )
    for candidate in remaining:
        if len(selected) >= MAX_MARKET_DOCUMENTS:
            break
        key = _canonical_url(str(candidate.url))
        if key not in selected_urls:
            selected.append(unique_results[key])
            selected_urls.add(key)

    return selected[:MAX_MARKET_DOCUMENTS]


def _hostname(url: str) -> str:
    return (urlsplit(url).hostname or "").removeprefix("www.").casefold()


def _same_domain(left: str, right: str) -> bool:
    left_host = _hostname(left)
    right_host = _hostname(right)
    return bool(
        left_host
        and right_host
        and (
            left_host == right_host
            or left_host.endswith(f".{right_host}")
            or right_host.endswith(f".{left_host}")
        )
    )


def classify_market_source(
    url: str,
    title: str,
    official_website: str | None,
) -> MarketSourceType:
    hostname = _hostname(url)
    searchable = f"{hostname} {title}".casefold()

    if official_website and _same_domain(url, official_website):
        return MarketSourceType.COMPANY_OFFICIAL
    if hostname.endswith(".gov") or ".gov." in hostname:
        return MarketSourceType.GOVERNMENT
    if any(
        marker in searchable
        for marker in (
            "industry report",
            "market report",
            "market research",
            "researchandmarkets",
            "grandviewresearch",
            "mordorintelligence",
            "kenresearch",
            "statista",
        )
    ):
        return MarketSourceType.INDUSTRY_REPORT
    if any(
        marker in searchable
        for marker in (
            "reuters",
            "bloomberg",
            "business-standard",
            "economictimes",
            "timesofindia",
            "news",
        )
    ):
        return MarketSourceType.NEWS
    if "official" in title.casefold():
        return MarketSourceType.COMPETITOR_OFFICIAL
    return MarketSourceType.OTHER


class MarketEvidenceCollector:
    def __init__(
        self,
        provider: MarketResearchProvider | None = None,
    ) -> None:
        self.provider = provider or TavilyProvider()

    async def collect(
        self,
        request: MarketEvidenceCollectionInput,
    ) -> MarketEvidenceCollectionOutput:
        queries = build_market_search_queries(request)
        search_responses = await asyncio.gather(
            *(
                self.provider.search(
                    TavilySearchRequest(
                        query=query,
                        search_depth="basic",
                        max_results=5,
                    )
                )
                for query in queries
            )
        )
        selected_results = select_market_results(
            search_responses,
            company_name=request.company_evidence.company_identity.name,
        )

        total_credits = sum(
            response.usage.credits_used
            for response in search_responses
        )
        total_response_time = sum(
            response.usage.response_time_seconds
            for response in search_responses
        )
        searched_results = sum(
            len(response.results)
            for response in search_responses
        )

        if not selected_results:
            return self._build_output(
                request=request,
                queries=queries,
                search_requests=len(search_responses),
                searched_results=searched_results,
                selected_urls=0,
                extract_response=None,
                documents=[],
                credits_used=total_credits,
                response_time_seconds=total_response_time,
                warnings=[
                    "No relevant market sources were found. Agent 3 was not run."
                ],
            )

        extraction_query = _compact_query(
            request.company_evidence.company_identity.name,
            request.company_evidence.company_identity.industry
            or "business sector",
            "competitors market trends entry barriers market share",
        )
        extract_response = await self.provider.extract(
            TavilyExtractRequest(
                urls=[result.url for result in selected_results],
                query=extraction_query,
                extract_depth="basic",
                chunks_per_source=3,
            )
        )
        total_credits += extract_response.usage.credits_used
        total_response_time += (
            extract_response.usage.response_time_seconds
        )

        metadata_by_url = {
            _canonical_url(str(result.url)): result
            for result in selected_results
        }
        official_website = (
            request.company_evidence.company_identity.official_website
        )
        market_documents: list[MarketSourceDocument] = []
        for extracted in extract_response.documents:
            content = extracted.content[:MAX_DOCUMENT_CHARACTERS].strip()
            if not content:
                continue

            metadata = metadata_by_url.get(
                _canonical_url(str(extracted.url))
            )
            title = metadata.title if metadata else "Market source"
            market_documents.append(
                MarketSourceDocument(
                    source_id=(
                        f"market_source_{len(market_documents) + 1}"
                    ),
                    title=title,
                    url=extracted.url,
                    source_type=classify_market_source(
                        str(extracted.url),
                        title,
                        official_website,
                    ),
                    content=content,
                )
            )

        warnings: list[str] = []
        if extract_response.failed_documents:
            warnings.append(
                f"{len(extract_response.failed_documents)} selected source(s) "
                "could not be extracted."
            )
        if not market_documents:
            warnings.append(
                "No selected source returned usable content. Agent 3 was not run."
            )

        return self._build_output(
            request=request,
            queries=queries,
            search_requests=len(search_responses),
            searched_results=searched_results,
            selected_urls=len(selected_results),
            extract_response=extract_response,
            documents=market_documents,
            credits_used=total_credits,
            response_time_seconds=total_response_time,
            warnings=warnings,
        )

    @staticmethod
    def _build_output(
        *,
        request: MarketEvidenceCollectionInput,
        queries: list[str],
        search_requests: int,
        searched_results: int,
        selected_urls: int,
        extract_response: TavilyExtractResponse | None,
        documents: list[MarketSourceDocument],
        credits_used: float,
        response_time_seconds: float,
        warnings: list[str],
    ) -> MarketEvidenceCollectionOutput:
        failed_sources = (
            [
                FailedMarketSource(
                    url=item.url,
                    error=item.error,
                )
                for item in extract_response.failed_documents
            ]
            if extract_response
            else []
        )

        return MarketEvidenceCollectionOutput(
            status=(
                MarketCollectionStatus.COMPLETED
                if documents
                else MarketCollectionStatus.INSUFFICIENT_SOURCES
            ),
            company_name=(
                request.company_evidence.company_identity.name
            ),
            search_queries=queries,
            market_documents=documents,
            failed_sources=failed_sources,
            usage=MarketCollectorUsage(
                search_requests=search_requests,
                extraction_requests=1 if extract_response else 0,
                searched_results=searched_results,
                selected_urls=selected_urls,
                extracted_documents=len(documents),
                credits_used=credits_used,
                response_time_seconds=response_time_seconds,
            ),
            warnings=warnings,
        )
