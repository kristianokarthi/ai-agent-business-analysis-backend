import logging
from dataclasses import dataclass
from typing import Generic, TypeVar

from groq import AsyncGroq
from pydantic import BaseModel

from app.core.config import settings
from app.schemas.llm import LLMUsage


ResponseModel = TypeVar(
    "ResponseModel",
    bound=BaseModel,
)

logger = logging.getLogger("uvicorn.error")


@dataclass
class StructuredLLMResult(Generic[ResponseModel]):
    data: ResponseModel
    usage: LLMUsage


class GroqProvider:
    def __init__(
        self,
        model: str | None = None,
    ) -> None:
        if not settings.groq_api_key:
            raise ValueError(
                "GROQ_API_KEY is missing from environment variables."
            )

        self.client = AsyncGroq(
            api_key=settings.groq_api_key,
        )

        self.model = model or settings.groq_model

    async def generate_structured(
        self,
        agent_name: str,
        system_prompt: str,
        user_prompt: str,
        response_model: type[ResponseModel],
        max_tokens: int = 1200,
        temperature: float = 0.1,
    ) -> StructuredLLMResult[ResponseModel]:

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            temperature=temperature,
            max_completion_tokens=max_tokens,
            reasoning_effort="low",
            include_reasoning=False,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": response_model.__name__.lower(),
                    "strict": True,
                    "schema": response_model.model_json_schema(),
                },
            },
        )

        content = response.choices[0].message.content

        if not content:
            raise ValueError(
                "Groq returned an empty response."
            )

        parsed_data = response_model.model_validate_json(
            content,
        )

        api_usage = response.usage

        usage = LLMUsage(
            agent_name=agent_name,
            provider="groq",
            model=self.model,
            input_tokens=api_usage.prompt_tokens if api_usage else 0,
            output_tokens=api_usage.completion_tokens if api_usage else 0,
            total_tokens=api_usage.total_tokens if api_usage else 0,
        )

        logger.info(
            "LLM usage | agent=%s | model=%s | "
            "input=%d | output=%d | total=%d",
            usage.agent_name,
            usage.model,
            usage.input_tokens,
            usage.output_tokens,
            usage.total_tokens,
        )

        return StructuredLLMResult(
            data=parsed_data,
            usage=usage,
        )