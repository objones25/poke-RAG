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


class JudgeScore(BaseModel):
    accuracy: int = Field(ge=0, le=100, description="Factual correctness vs reference material")
    groundedness: int = Field(
        ge=0, le=100, description="Grounded in retrieved context, not hallucinated"
    )
    domain_correctness: int = Field(
        ge=0, le=100, description="Pokémon domain knowledge applied correctly"
    )

    @property
    def total(self) -> int:
        return self.accuracy + self.groundedness + self.domain_correctness


class CandidateResponse(BaseModel):
    response: str
    judge_score: JudgeScore | None = None


class DPODatapoint(BaseModel):
    prompt: list[dict[str, str]]
    chosen: list[dict[str, str]]
    rejected: list[dict[str, str]]
