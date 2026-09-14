import asyncio
import ipaddress
import socket
from html.parser import HTMLParser
from urllib.parse import urljoin, urlparse

import httpx

from app.schemas.fact_finder import SourceDocument


MAX_REDIRECTS = 3
MAX_SOURCE_CHARACTERS = 20_000
REQUEST_TIMEOUT_SECONDS = 15.0


class SourceCollectionError(Exception):
    pass


class _PageTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._ignored_depth = 0
        self._in_title = False
        self.title_parts: list[str] = []
        self.text_parts: list[str] = []

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        if tag in {"script", "style", "noscript", "svg"}:
            self._ignored_depth += 1
        elif tag == "title":
            self._in_title = True

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript", "svg"}:
            self._ignored_depth = max(0, self._ignored_depth - 1)
        elif tag == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        text = " ".join(data.split())
        if not text:
            return
        if self._in_title:
            self.title_parts.append(text)
        if self._ignored_depth == 0:
            self.text_parts.append(text)


async def _ensure_public_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise SourceCollectionError("The official website URL is invalid.")

    try:
        address_info = await asyncio.to_thread(
            socket.getaddrinfo,
            parsed.hostname,
            parsed.port,
            type=socket.SOCK_STREAM,
        )
    except socket.gaierror as error:
        raise SourceCollectionError(
            "The official website hostname could not be resolved."
        ) from error

    addresses = {
        item[4][0]
        for item in address_info
    }
    if not addresses or any(
        not ipaddress.ip_address(address).is_global
        for address in addresses
    ):
        raise SourceCollectionError(
            "The official website must resolve to a public internet address."
        )


async def collect_official_website(
    website_url: str,
) -> SourceDocument:
    current_url = website_url

    async with httpx.AsyncClient(
        timeout=REQUEST_TIMEOUT_SECONDS,
        headers={
            "User-Agent": (
                "VentureLensAI/0.1 "
                "(business research source collector)"
            ),
            "Accept": "text/html,text/plain",
        },
        follow_redirects=False,
    ) as client:
        for redirect_count in range(MAX_REDIRECTS + 1):
            await _ensure_public_url(current_url)

            try:
                response = await client.get(current_url)
            except httpx.HTTPError as error:
                raise SourceCollectionError(
                    "The official website could not be reached."
                ) from error

            if response.is_redirect:
                location = response.headers.get("location")
                if not location or redirect_count == MAX_REDIRECTS:
                    raise SourceCollectionError(
                        "The official website redirected too many times."
                    )
                current_url = urljoin(current_url, location)
                continue

            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as error:
                raise SourceCollectionError(
                    "The official website rejected the research request "
                    f"with status {response.status_code}."
                ) from error

            content_type = response.headers.get(
                "content-type",
                "",
            ).lower()
            if not any(
                supported in content_type
                for supported in ("text/html", "text/plain")
            ):
                raise SourceCollectionError(
                    "The official website did not return a readable page."
                )

            parser = _PageTextParser()
            parser.feed(response.text)
            page_text = " ".join(parser.text_parts)
            page_text = page_text[:MAX_SOURCE_CHARACTERS].strip()

            if not page_text:
                raise SourceCollectionError(
                    "No readable text was found on the official website."
                )

            title = " ".join(parser.title_parts).strip()
            return SourceDocument(
                source_id="source_1",
                title=title or "Official company website",
                url=current_url,
                content=page_text,
            )

    raise SourceCollectionError(
        "The official website could not be collected."
    )
