import logging
from typing import TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from app.core.config import settings
from app.llm.types import StructuredLLMResult
from app.schemas.llm import LLMUsage


ResponseModel = TypeVar("ResponseModel", bound=BaseModel)

OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"
REQUEST_TIMEOUT_SECONDS = 120.0

logger = logging.getLogger("uvicorn.error")


class OpenRouterProviderError(Exception):
    pass


class OpenRouterConfigurationError(OpenRouterProviderError):
    pass


class OpenRouterCreditError(OpenRouterProviderError):
    pass


class OpenRouterRequestError(OpenRouterProviderError):
    pass


class OpenRouterRateLimitError(OpenRouterProviderError):
    pass


class OpenRouterInvalidResponseError(OpenRouterProviderError):
    pass


class OpenRouterTruncatedResponseError(OpenRouterProviderError):
    pass


class OpenRouterServiceError(OpenRouterProviderError):
    pass


class OpenRouterProvider:
    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.api_key = api_key or settings.openrouter_api_key
        self.model = model or settings.openrouter_model
        self.transport = transport

        if not self.api_key:
            raise OpenRouterConfigurationError(
                "OPENROUTER_API_KEY is missing from environment variables."
            )

    async def generate_structured(
        self,
        agent_name: str,
        system_prompt: str,
        user_prompt: str,
        response_model: type[ResponseModel],
        max_tokens: int = 6_000,
        temperature: float = 0,
    ) -> StructuredLLMResult[ResponseModel]:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "reasoning": {
                "enabled": False,
            },
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": response_model.__name__.lower(),
                    "strict": True,
                    "schema": response_model.model_json_schema(),
                },
            },
            "provider": {
                "require_parameters": True,
            },
        }

        try:
            async with httpx.AsyncClient(
                base_url=OPENROUTER_API_BASE,
                timeout=REQUEST_TIMEOUT_SECONDS,
                transport=self.transport,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
            ) as client:
                response = await client.post(
                    "/chat/completions",
                    json=payload,
                )
        except httpx.TimeoutException as error:
            raise OpenRouterServiceError(
                "OpenRouter did not respond before the request timed out."
            ) from error
        except httpx.HTTPError as error:
            raise OpenRouterServiceError(
                "OpenRouter could not be reached."
            ) from error

        if response.status_code in {401, 403}:
            logger.error(
                "OpenRouter authentication error | status=%d | body=%s",
                response.status_code,
                response.text[:1000],
            )
            raise OpenRouterConfigurationError(
                "The OpenRouter API key is invalid or lacks model access."
            )
        if response.status_code == 402:
            raise OpenRouterCreditError(
                "The OpenRouter account does not have enough credit."
            )
        if response.status_code == 429:
            raise OpenRouterRateLimitError(
                "The OpenRouter request limit has been reached."
            )
        if 400 <= response.status_code < 500:
            logger.error(
                "OpenRouter request rejected | status=%d | body=%s",
                response.status_code,
                response.text[:1000],
            )
            raise OpenRouterRequestError(
                "OpenRouter rejected the generation request."
            )
        if response.status_code >= 500:
            raise OpenRouterServiceError(
                "OpenRouter is temporarily unavailable."
            )

        try:
            body = response.json()
            choice = body["choices"][0]
            content = choice["message"]["content"]
            finish_reason = choice.get("finish_reason")
        except (KeyError, IndexError, TypeError, ValueError) as error:
            raise OpenRouterInvalidResponseError(
                "OpenRouter returned an invalid response envelope."
            ) from error

        api_usage = body.get("usage") or {}
        completion_details = (
            api_usage.get("completion_tokens_details") or {}
        )
        usage = LLMUsage(
            agent_name=agent_name,
            provider="openrouter",
            model=body.get("model") or self.model,
            input_tokens=api_usage.get("prompt_tokens", 0),
            output_tokens=api_usage.get("completion_tokens", 0),
            reasoning_tokens=completion_details.get("reasoning_tokens", 0),
            total_tokens=api_usage.get("total_tokens", 0),
        )

        if finish_reason in {"length", "max_tokens"}:
            logger.warning(
                "OpenRouter output truncated | agent=%s | model=%s | "
                "finish_reason=%s | output=%d | reasoning=%d",
                agent_name,
                usage.model,
                finish_reason,
                usage.output_tokens,
                usage.reasoning_tokens,
            )
            raise OpenRouterTruncatedResponseError(
                "OpenRouter stopped because the output-token budget was reached."
            )

        if not isinstance(content, str) or not content.strip():
            logger.warning(
                "OpenRouter returned empty content | agent=%s | model=%s | "
                "finish_reason=%s | output=%d | reasoning=%d",
                agent_name,
                usage.model,
                finish_reason,
                usage.output_tokens,
                usage.reasoning_tokens,
            )
            raise OpenRouterInvalidResponseError(
                "OpenRouter returned empty structured content."
            )

        try:
            parsed = response_model.model_validate_json(content)
        except ValidationError as error:
            logger.warning(
                "OpenRouter structured validation failed | agent=%s | "
                "model=%s | finish_reason=%s | content_chars=%d | errors=%s",
                agent_name,
                usage.model,
                finish_reason,
                len(content),
                error.errors(include_input=False),
            )
            raise OpenRouterInvalidResponseError(
                "OpenRouter returned content that did not match the report schema."
            ) from error

        logger.info(
            "LLM usage | agent=%s | provider=openrouter | model=%s | "
            "input=%d | output=%d | reasoning=%d | total=%d",
            usage.agent_name,
            usage.model,
            usage.input_tokens,
            usage.output_tokens,
            usage.reasoning_tokens,
            usage.total_tokens,
        )

        return StructuredLLMResult(data=parsed, usage=usage)
