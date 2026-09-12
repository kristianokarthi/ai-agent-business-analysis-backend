from pydantic import BaseModel


class LLMUsage(BaseModel):
    agent_name: str
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    total_tokens: int