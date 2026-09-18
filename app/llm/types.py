from dataclasses import dataclass
from typing import Generic, TypeVar

from pydantic import BaseModel

from app.schemas.llm import LLMUsage


ResponseModel = TypeVar("ResponseModel", bound=BaseModel)


@dataclass
class StructuredLLMResult(Generic[ResponseModel]):
    data: ResponseModel
    usage: LLMUsage
