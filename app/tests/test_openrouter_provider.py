import asyncio
import json

import httpx
import pytest
from pydantic import Field

from app.llm.openrouter_provider import (
    OpenRouterConfigurationError,
    OpenRouterCreditError,
    OpenRouterInvalidResponseError,
    OpenRouterProvider,
    OpenRouterRateLimitError,
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
        assert request.url.path == "/api/v1/chat/completions"
        assert request.headers["Authorization"] == "Bearer test-key"

        payload = json.loads(request.content)
        assert payload["model"] == "nvidia/nemotron-3-super-120b-a12b"
        assert payload["max_tokens"] == 2500
        assert payload["provider"]["require_parameters"] is True
        response_format = payload["response_format"]
        assert response_format["type"] == "json_schema"
        assert response_format["json_schema"]["strict"] is True
        assert response_format["json_schema"]["schema"]["type"] == "object"

        return json_response(
            {
                "id": "generation-test",
                "model": "nvidia/nemotron-3-super-120b-a12b",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": json.dumps({"summary": "Done"}),
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1200,
                    "completion_tokens": 1800,
                    "total_tokens": 3200,
                    "completion_tokens_details": {
                        "reasoning_tokens": 200,
                    },
                },
            }
        )

    provider = OpenRouterProvider(
        api_key="test-key",
        model="nvidia/nemotron-3-super-120b-a12b",
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
    assert result.usage.provider == "openrouter"
    assert result.usage.input_tokens == 1200
    assert result.usage.output_tokens == 1800
    assert result.usage.reasoning_tokens == 200
    assert result.usage.total_tokens == 3200


def test_missing_api_key_is_rejected(monkeypatch):
    monkeypatch.setattr(
        "app.llm.openrouter_provider.settings.openrouter_api_key",
        "",
    )
    with pytest.raises(
        OpenRouterConfigurationError,
        match="OPENROUTER_API_KEY",
    ):
        OpenRouterProvider(api_key="")


@pytest.mark.parametrize(
    ("status_code", "error_type"),
    [
        (402, OpenRouterCreditError),
        (429, OpenRouterRateLimitError),
    ],
)
def test_provider_limits_have_specific_errors(status_code, error_type):
    async def handler(_: httpx.Request) -> httpx.Response:
        return json_response(
            {"error": {"message": "limit reached"}},
            status_code=status_code,
        )

    provider = OpenRouterProvider(
        api_key="test-key",
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(error_type):
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
                "choices": [
                    {"message": {"content": "not-json"}},
                ],
            }
        )

    provider = OpenRouterProvider(
        api_key="test-key",
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(OpenRouterInvalidResponseError):
        asyncio.run(
            provider.generate_structured(
                agent_name="report_strategist",
                system_prompt="System",
                user_prompt="User",
                response_model=SampleResponse,
            )
        )
