import json
import re

from app.llm.openrouter_provider import OpenRouterProvider
from app.llm.types import StructuredLLMResult
from app.prompts.rag_answer import GROUNDED_ANSWER_SYSTEM_PROMPT
from app.schemas.rag import (
    ChatMessage,
    GroundedAnswerDraft,
    GroundedAnswerStatus,
    SemanticSearchMatch,
)


class InvalidGroundedAnswerError(Exception):
    pass


class GroundedAnswerAgent:
    def __init__(self, provider: OpenRouterProvider) -> None:
        self.provider = provider

    async def run(
        self,
        question: str,
        matches: list[SemanticSearchMatch],
        conversation_history: list[ChatMessage],
    ) -> StructuredLLMResult[GroundedAnswerDraft]:
        retrieved_chunks = [
            {
                "chunk_id": match.chunk.chunk_id,
                "section": match.chunk.section.value,
                "title": match.chunk.title,
                "content": match.chunk.content,
                "evidence_ids": match.chunk.evidence_ids,
            }
            for match in matches
        ]
        result = await self.provider.generate_structured(
            agent_name="rag_grounded_answer",
            system_prompt=GROUNDED_ANSWER_SYSTEM_PROMPT,
            user_prompt=json.dumps(
                {
                    "question": question,
                    "conversation_history": [
                        message.model_dump(mode="json")
                        for message in conversation_history
                    ],
                    "retrieved_chunks": retrieved_chunks,
                },
                ensure_ascii=False,
            ),
            response_model=GroundedAnswerDraft,
            max_tokens=700,
            temperature=0,
        )

        allowed_ids = {
            match.chunk.chunk_id
            for match in matches
        }
        returned_ids = result.data.supporting_chunk_ids
        if len(returned_ids) != len(set(returned_ids)):
            raise InvalidGroundedAnswerError(
                "The answer repeated a supporting chunk ID."
            )
        if not set(returned_ids).issubset(allowed_ids):
            raise InvalidGroundedAnswerError(
                "The answer cited a chunk that was not retrieved."
            )
        if (
            result.data.status == GroundedAnswerStatus.ANSWERED
            and not returned_ids
        ):
            raise InvalidGroundedAnswerError(
                "An answered response must cite at least one retrieved chunk."
            )
        if (
            result.data.status == GroundedAnswerStatus.OUT_OF_SCOPE
            and returned_ids
        ):
            raise InvalidGroundedAnswerError(
                "An out-of-scope response cannot cite report chunks."
            )

        prohibited_recommendation = re.compile(
            r"\b(?:recommend(?:ation)?\s+(?:is\s+)?(?:to\s+)?)?"
            r"(?:buy|sell|hold)\s+(?:the\s+)?(?:stock|shares?)\b",
            re.IGNORECASE,
        )
        if prohibited_recommendation.search(result.data.answer):
            raise InvalidGroundedAnswerError(
                "The answer returned a direct stock-trading instruction."
            )

        return result
