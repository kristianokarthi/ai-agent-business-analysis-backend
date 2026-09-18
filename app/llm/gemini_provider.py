import logging
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from app.core.config import settings
from app.llm.types import StructuredLLMResult
from app.schemas.llm import LLMUsage


ResponseModel = TypeVar("ResponseModel", bound=BaseModel)

GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"
REQUEST_TIMEOUT_SECONDS = 90.0

logger = logging.getLogger("uvicorn.error")


class GeminiProviderError(Exception):
    pass


class GeminiConfigurationError(GeminiProviderError):
    pass


class GeminiRequestError(GeminiProviderError):
    pass


class GeminiRateLimitError(GeminiProviderError):
    pass


class GeminiInvalidResponseError(GeminiProviderError):
    pass


class GeminiServiceError(GeminiProviderError):
    pass


def _gemini_json_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Remove Pydantic keywords unsupported by Gemini's schema subset."""
    unsupported = {
        "default",
        "examples",
        "minLength",
        "maxLength",
        "pattern",
    }
    return {
        key: (
            _gemini_json_schema(value)
            if isinstance(value, dict)
            else [
                _gemini_json_schema(item) if isinstance(item, dict) else item
                for item in value
            ]
            if isinstance(value, list)
            else value
        )
        for key, value in schema.items()
        if key not in unsupported
    }


class GeminiProvider:
    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.api_key = api_key or settings.gemini_api_key
        self.model = model or settings.gemini_model
        self.transport = transport

        if not self.api_key:
            raise GeminiConfigurationError(
                "GEMINI_API_KEY is missing from environment variables."
            )

    async def generate_structured(
        self,
        agent_name: str,
        system_prompt: str,
        user_prompt: str,
        response_model: type[ResponseModel],
        max_tokens: int = 3_500,
        temperature: float = 0,
    ) -> StructuredLLMResult[ResponseModel]:
        payload = {
            "systemInstruction": {
                "parts": [{"text": system_prompt}],
            },
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": user_prompt}],
                }
            ],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
                "responseMimeType": "application/json",
                "responseJsonSchema": _gemini_json_schema(
                    response_model.model_json_schema()
                ),
            },
        }

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
                    f"/models/{self.model}:generateContent",
                    json=payload,
                )
        except httpx.TimeoutException as error:
            raise GeminiServiceError(
                "Gemini did not respond before the request timed out."
            ) from error
        except httpx.HTTPError as error:
            raise GeminiServiceError(
                "Gemini could not be reached."
            ) from error

        if response.status_code in {401, 403}:
            raise GeminiConfigurationError(
                "The Gemini API key is invalid or lacks access to the model."
            )
        if response.status_code == 429:
            raise GeminiRateLimitError(
                "The Gemini project rate or daily limit has been reached."
            )
        if 400 <= response.status_code < 500:
            raise GeminiRequestError(
                "Gemini rejected the generation request."
            )
        if response.status_code >= 500:
            raise GeminiServiceError(
                "Gemini is temporarily unavailable."
            )

        try:
            body = response.json()
            candidate = body["candidates"][0]
            parts = candidate["content"]["parts"]
            content = "".join(
                part.get("text", "")
                for part in parts
                if isinstance(part, dict)
            )
            if not content:
                raise KeyError("empty candidate content")
            parsed = response_model.model_validate_json(content)
        except (
            KeyError,
            IndexError,
            TypeError,
            ValueError,
            ValidationError,
        ) as error:
            raise GeminiInvalidResponseError(
                "Gemini returned an invalid structured response."
            ) from error

        api_usage = body.get("usageMetadata", {})
        usage = LLMUsage(
            agent_name=agent_name,
            provider="google",
            model=body.get("modelVersion") or self.model,
            input_tokens=api_usage.get("promptTokenCount", 0),
            output_tokens=api_usage.get("candidatesTokenCount", 0),
            reasoning_tokens=api_usage.get("thoughtsTokenCount", 0),
            total_tokens=api_usage.get("totalTokenCount", 0),
        )

        logger.info(
            "LLM usage | agent=%s | provider=google | model=%s | "
            "input=%d | output=%d | reasoning=%d | total=%d",
            usage.agent_name,
            usage.model,
            usage.input_tokens,
            usage.output_tokens,
            usage.reasoning_tokens,
            usage.total_tokens,
        )

        return StructuredLLMResult(data=parsed, usage=usage)
