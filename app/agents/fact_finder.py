from app.llm.groq_provider import (
    GroqProvider,
    StructuredLLMResult,
)
from app.prompts.fact_finder import build_fact_finder_prompt
from app.schemas.fact_finder import (
    FactFinderInput,
    FactFinderOutput,
)


class FactFinderAgent:
    def __init__(
        self,
        provider: GroqProvider | None = None,
    ) -> None:
        self.provider = provider or GroqProvider()

    async def run(
        self,
        agent_input: FactFinderInput,
    ) -> StructuredLLMResult[FactFinderOutput]:

        system_prompt = build_fact_finder_prompt(
            agent_input.purpose,
        )

        user_prompt = (
            "Analyse the following company research input.\n\n"
            f"{agent_input.model_dump_json(indent=2)}"
        )

        return await self.provider.generate_structured(
            agent_name="fact_finder",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=FactFinderOutput,
            max_tokens=2000,
            temperature=0.1,
        )