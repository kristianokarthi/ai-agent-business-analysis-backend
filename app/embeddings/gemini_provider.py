import logging
import math
from dataclasses import dataclass
from typing import Any

import httpx

from app.core.config import settings
from app.schemas.rag import ReportChunk


GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"
REQUEST_TIMEOUT_SECONDS = 30.0

logger = logging.getLogger("uvicorn.error")


class GeminiEmbeddingError(Exception):
    pass


class GeminiEmbeddingConfigurationError(GeminiEmbeddingError):
    pass


class GeminiEmbeddingRequestError(GeminiEmbeddingError):
    pass


class GeminiEmbeddingRateLimitError(GeminiEmbeddingError):
    pass


class GeminiEmbeddingServiceError(GeminiEmbeddingError):
    pass


@dataclass(frozen=True)
class EmbeddingBatch:
    vectors: list[list[float]]
    input_tokens: int


def _normalize(vector: list[float]) -> list[float]:
    magnitude = math.sqrt(sum(value * value for value in vector))
    if magnitude == 0:
        raise GeminiEmbeddingServiceError(
            "Gemini returned an empty embedding vector."
        )
    return [value / magnitude for value in vector]


class GeminiEmbeddingProvider:
    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        dimensions: int | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.api_key = (
            settings.gemini_api_key if api_key is None else api_key
        )
        self.model = (
            settings.gemini_embedding_model if model is None else model
        )
        self.dimensions = (
            settings.gemini_embedding_dimensions
            if dimensions is None
            else dimensions
        )
        self.transport = transport

        if not self.api_key:
            raise GeminiEmbeddingConfigurationError(
                "GEMINI_API_KEY is missing from environment variables."
            )
        if not 1 <= self.dimensions <= 3072:
            raise GeminiEmbeddingConfigurationError(
                "GEMINI_EMBEDDING_DIMENSIONS must be between 1 and 3072."
            )

    async def _embed_requests(
        self,
        requests: list[dict[str, Any]],
    ) -> EmbeddingBatch:
        try:
            async with httpx.AsyncClient(
                base_url=GEMINI_API_BASE,
                timeout=REQUEST_TIMEOUT_SECONDS,
                transport=self.transport,
                headers={
                    "x-goog-api-key": self.api_key,
                    "Content-Type": "application/json",
                },
            ) as client:
                response = await client.post(
                    f"/models/{self.model}:batchEmbedContents",
                    json={"requests": requests},
                )
        except httpx.TimeoutException as error:
            raise GeminiEmbeddingServiceError(
                "Gemini embeddings timed out."
            ) from error
        except httpx.HTTPError as error:
            raise GeminiEmbeddingServiceError(
                "Gemini embeddings could not be reached."
            ) from error

        if response.status_code in {401, 403}:
            raise GeminiEmbeddingConfigurationError(
                "The Gemini API key is invalid or lacks embedding access."
            )
        if response.status_code == 429:
            raise GeminiEmbeddingRateLimitError(
                "The Gemini embedding rate limit has been reached."
            )
        if 400 <= response.status_code < 500:
            logger.error(
                "Gemini embedding request rejected | status=%d | body=%s",
                response.status_code,
                response.text[:1000],
            )
            raise GeminiEmbeddingRequestError(
                "Gemini rejected the embedding request."
            )
        if response.status_code >= 500:
            raise GeminiEmbeddingServiceError(
                "Gemini embeddings are temporarily unavailable."
            )

        try:
            body: dict[str, Any] = response.json()
            raw_embeddings = body["embeddings"]
            vectors = [
                _normalize([float(value) for value in item["values"]])
                for item in raw_embeddings
            ]
            usage = body.get("usageMetadata") or {}
            input_tokens = int(
                usage.get("promptTokenCount")
                or usage.get("totalTokenCount")
                or 0
            )
        except (KeyError, TypeError, ValueError) as error:
            raise GeminiEmbeddingServiceError(
                "Gemini returned an invalid embedding response."
            ) from error

        if len(vectors) != len(requests):
            raise GeminiEmbeddingServiceError(
                "Gemini returned a different number of embeddings than requested."
            )
        if any(len(vector) != self.dimensions for vector in vectors):
            raise GeminiEmbeddingServiceError(
                "Gemini returned an unexpected embedding dimension."
            )

        return EmbeddingBatch(
            vectors=vectors,
            input_tokens=input_tokens,
        )

    async def embed_documents(
        self,
        chunks: list[ReportChunk],
    ) -> EmbeddingBatch:
        model_name = f"models/{self.model}"
        requests = [
            {
                "model": model_name,
                "content": {
                    "parts": [
                        {
                            "text": (
                                f"Section: {chunk.section.value}\n"
                                f"Title: {chunk.title}\n"
                                f"Content: {chunk.content}"
                            )
                        }
                    ]
                },
                "taskType": "RETRIEVAL_DOCUMENT",
                "title": chunk.title,
                "outputDimensionality": self.dimensions,
            }
            for chunk in chunks
        ]
        batch = await self._embed_requests(requests)
        logger.info(
            "Embedding usage | provider=gemini | model=%s | task=document | "
            "items=%d | dimensions=%d | input_tokens=%d | requests=1",
            self.model,
            len(chunks),
            self.dimensions,
            batch.input_tokens,
        )
        return batch

    async def embed_question(self, question: str) -> EmbeddingBatch:
        batch = await self._embed_requests(
            [
                {
                    "model": f"models/{self.model}",
                    "content": {"parts": [{"text": question}]},
                    "taskType": "QUESTION_ANSWERING",
                    "outputDimensionality": self.dimensions,
                }
            ]
        )
        logger.info(
            "Embedding usage | provider=gemini | model=%s | task=question | "
            "items=1 | dimensions=%d | input_tokens=%d | requests=1",
            self.model,
            self.dimensions,
            batch.input_tokens,
        )
        return batch
