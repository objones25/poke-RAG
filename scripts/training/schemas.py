from __future__ import annotations

from pydantic import BaseModel, Field


class GeminiQAPair(BaseModel):
    question: str = Field(description="A natural question answerable from the provided chunk")
    answer: str = Field(description="A concise, grounded answer based only on the chunk")


class SFTMessage(BaseModel):
    role: str
    content: str


class SFTDatapoint(BaseModel):
    messages: list[SFTMessage]
