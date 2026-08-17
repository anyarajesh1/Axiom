from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

VerdictLabel = Literal[
    "supported",
    "contradicted",
    "insufficient_evidence",
]


class Claim(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    text: str = Field(min_length=1)
    source_span: str | None = None


class ExtractedClaim(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1)
    source_span: str = Field(min_length=1)


class ClaimExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claims: list[ExtractedClaim]


class AnalyzeClaimsRequest(BaseModel):
    text: str = Field(min_length=1, max_length=20_000)


class AnalyzeClaimsResponse(BaseModel):
    submission_id: UUID
    claims: list[Claim]


class Evidence(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    text: str = Field(min_length=1)
    source_url: str = Field(min_length=1)
    score: float = Field(ge=0, le=1)


class Verdict(BaseModel):
    claim_id: UUID
    label: VerdictLabel
    confidence: float = Field(ge=0, le=1)
    evidence_ids: list[UUID] = Field(default_factory=list)
    explanation: str = Field(min_length=1)


class PipelineState(BaseModel):
    input_text: str = Field(min_length=1)
    claims: list[Claim] = Field(default_factory=list)
    evidence_by_claim: dict[UUID, list[Evidence]] = Field(default_factory=dict)
    verdicts: list[Verdict] = Field(default_factory=list)
