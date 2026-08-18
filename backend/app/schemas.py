from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

VerdictLabel = Literal[
    "supported",
    "contradicted",
    "insufficient_evidence",
]
EntailmentLabel = Literal["entailment", "contradiction", "neutral"]


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
    source_name: str = Field(default="Unknown source", min_length=1)
    source_url: str = Field(min_length=1)
    category: str = Field(default="uncategorized", min_length=1)
    score: float = Field(ge=0, le=1)
    reranker_score: float | None = Field(default=None, ge=0, le=1)
    combined_score: float | None = Field(default=None, ge=0, le=1)


class EntailmentResult(BaseModel):
    evidence_id: UUID
    label: EntailmentLabel
    confidence: float = Field(ge=0, le=1)
    scores: dict[EntailmentLabel, float]


class ContradictionSummary(BaseModel):
    claim_id: UUID
    support_count: int = Field(ge=0)
    contradiction_count: int = Field(ge=0)
    neutral_count: int = Field(ge=0)
    has_conflict: bool


class AnalyzeEvidenceRequest(BaseModel):
    claim: str = Field(min_length=1, max_length=4_000)
    limit: int = Field(default=5, ge=1, le=10)


class AnalyzeEvidenceResponse(BaseModel):
    claim: str
    evidence: list[Evidence]
    external_search_used: bool


class VerifyClaimRequest(BaseModel):
    claim: str = Field(min_length=1, max_length=4_000)
    candidate_limit: int = Field(default=8, ge=1, le=12)
    evidence_limit: int = Field(default=5, ge=1, le=8)


class VerifyClaimResponse(BaseModel):
    claim: str
    evidence: list[Evidence]
    entailments: list[EntailmentResult]
    external_search_used: bool


class Verdict(BaseModel):
    claim_id: UUID
    label: VerdictLabel
    confidence: float = Field(ge=0, le=1)
    evidence_ids: list[UUID] = Field(default_factory=list)
    explanation: str = Field(min_length=1)


class RefereeDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: VerdictLabel
    confidence: float = Field(ge=0, le=1)
    evidence_ids: list[str]
    explanation: str = Field(min_length=1, max_length=400)

    @field_validator("explanation")
    @classmethod
    def explanation_under_forty_words(cls, value: str) -> str:
        if len(value.split()) > 40:
            raise ValueError("Explanation must not exceed 40 words.")
        return value


class AnalyzeRequest(BaseModel):
    text: str = Field(min_length=1, max_length=20_000)


class AnalyzeResponse(BaseModel):
    submission_id: UUID
    claims: list[Claim]
    evidence_by_claim: dict[UUID, list[Evidence]]
    entailment_by_claim: dict[UUID, list[EntailmentResult]]
    contradictions_by_claim: dict[UUID, ContradictionSummary]
    verdicts: list[Verdict]


class PipelineState(BaseModel):
    input_text: str = Field(min_length=1)
    claims: list[Claim] = Field(default_factory=list)
    evidence_by_claim: dict[UUID, list[Evidence]] = Field(default_factory=dict)
    entailment_by_claim: dict[UUID, list[EntailmentResult]] = Field(
        default_factory=dict
    )
    contradictions_by_claim: dict[UUID, ContradictionSummary] = Field(
        default_factory=dict
    )
    verdicts: list[Verdict] = Field(default_factory=list)
