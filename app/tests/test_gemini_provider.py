import asyncio
import json

import httpx
import pytest
from pydantic import Field

from app.llm.gemini_provider import (
    GeminiConfigurationError,
    GeminiInvalidResponseError,
    GeminiProvider,
    GeminiRateLimitError,
)
from app.schemas.fact_finder import StrictSchema


class SampleResponse(StrictSchema):
    summary: str = Field(min_length=1, max_length=100)


def json_response(payload: dict, status_code: int = 200) -> httpx.Response:
    return httpx.Response(
        status_code,
        content=json.dumps(payload),
        headers={"Content-Type": "application/json"},
    )


def test_generate_structured_returns_data_and_usage():
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == (
            "/v1beta/models/gemini-2.5-flash:generateContent"
        )
        assert request.headers["x-goog-api-key"] == "test-key"
        payload = json.loads(request.content)
        config = payload["generationConfig"]
        assert config["responseMimeType"] == "application/json"
        assert config["maxOutputTokens"] == 2500
        assert "responseJsonSchema" not in config
        prompt = payload["contents"][0]["parts"][0]["text"]
        assert "Required JSON Schema" in prompt
        assert "minLength" not in prompt

        return json_response(
            {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {"text": json.dumps({"summary": "Done"})}
                            ]
                        },
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 1200,
                    "candidatesTokenCount": 1800,
                    "thoughtsTokenCount": 200,
                    "totalTokenCount": 3200,
                },
                "modelVersion": "gemini-2.5-flash-test",
            }
        )

    provider = GeminiProvider(
        api_key="test-key",
        model="gemini-2.5-flash",
        transport=httpx.MockTransport(handler),
    )
    result = asyncio.run(
        provider.generate_structured(
            agent_name="report_strategist",
            system_prompt="System",
            user_prompt="User",
            response_model=SampleResponse,
            max_tokens=2500,
        )
    )

    assert result.data.summary == "Done"
    assert result.usage.provider == "google"
    assert result.usage.input_tokens == 1200
    assert result.usage.output_tokens == 1800
    assert result.usage.reasoning_tokens == 200
    assert result.usage.total_tokens == 3200


def test_missing_api_key_is_rejected(monkeypatch):
    monkeypatch.setattr(
        "app.llm.gemini_provider.settings.gemini_api_key",
        "",
    )
    with pytest.raises(GeminiConfigurationError, match="GEMINI_API_KEY"):
        GeminiProvider(api_key="")


def test_rate_limit_has_specific_error():
    async def handler(_: httpx.Request) -> httpx.Response:
        return json_response(
            {"error": {"status": "RESOURCE_EXHAUSTED"}},
            status_code=429,
        )

    provider = GeminiProvider(
        api_key="test-key",
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(GeminiRateLimitError):
        asyncio.run(
            provider.generate_structured(
                agent_name="report_strategist",
                system_prompt="System",
                user_prompt="User",
                response_model=SampleResponse,
            )
        )


def test_invalid_structured_response_is_rejected():
    async def handler(_: httpx.Request) -> httpx.Response:
        return json_response(
            {
                "candidates": [
                    {"content": {"parts": [{"text": "not-json"}]}}
                ]
            }
        )

    provider = GeminiProvider(
        api_key="test-key",
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(GeminiInvalidResponseError):
        asyncio.run(
            provider.generate_structured(
                agent_name="report_strategist",
                system_prompt="System",
                user_prompt="User",
                response_model=SampleResponse,
            )
        )
