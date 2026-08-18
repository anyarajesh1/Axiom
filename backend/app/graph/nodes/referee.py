import json
import logging
from collections.abc import Callable
from uuid import UUID

from groq import Groq
from pydantic import ValidationError

from app.config import settings
from app.schemas import (
    Claim,
    ContradictionSummary,
    EntailmentResult,
    Evidence,
    RefereeDecision,
    Verdict,
)

logger = logging.getLogger(__name__)
RequestDecision = Callable[[str], str]

SYSTEM_PROMPT = """You are Axiom's evidence referee.

Decide whether the claim is supported, contradicted, or has insufficient
evidence using only the supplied evidence and ML scores. Do not use outside
knowledge. Treat low combined scores cautiously. If relevant sources conflict,
acknowledge that uncertainty. Cite evidence only by its supplied UUID.

Return JSON matching the provided schema. Keep the explanation at 40 words or
fewer. Never invent an evidence ID.
"""


class RefereeError(RuntimeError):
    """Raised when the referee cannot return a valid verdict."""


def build_referee_input(
    claim: Claim,
    evidence: list[Evidence],
    entailments: list[EntailmentResult],
    contradiction: ContradictionSummary,
) -> str:
    entailment_by_id = {result.evidence_id: result for result in entailments}
    evidence_payload = []
    for item in evidence:
        result = entailment_by_id.get(item.id)
        evidence_payload.append(
            {
                "id": str(item.id),
                "text": item.text,
                "source_name": item.source_name,
                "source_url": item.source_url,
                "combined_score": item.combined_score,
                "nli_label": result.label if result else "neutral",
                "nli_confidence": result.confidence if result else 0,
            }
        )

    return json.dumps(
        {
            "claim": claim.text,
            "conflict_summary": contradiction.model_dump(mode="json"),
            "ranked_evidence": evidence_payload,
        },
        separators=(",", ":"),
    )


def request_decision(prompt: str) -> str:
    with Groq(
        api_key=settings.GROQ_API_KEY,
        timeout=30.0,
        max_retries=2,
    ) as client:
        response = client.chat.completions.create(
            model=settings.GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "axiom_referee_decision",
                    "strict": True,
                    "schema": RefereeDecision.model_json_schema(),
                },
            },
            temperature=0,
            max_tokens=500,
        )

    content = response.choices[0].message.content
    if not content:
        raise RefereeError("Groq returned an empty referee decision.")
    return content


def fallback_verdict(claim: Claim, explanation: str) -> Verdict:
    return Verdict(
        claim_id=claim.id,
        label="insufficient_evidence",
        confidence=0,
        evidence_ids=[],
        explanation=explanation,
    )


def decide_verdict(
    claim: Claim,
    evidence: list[Evidence],
    entailments: list[EntailmentResult],
    contradiction: ContradictionSummary,
    request: RequestDecision | None = None,
) -> Verdict:
    if not evidence:
        return fallback_verdict(
            claim,
            "No sufficiently relevant evidence was available for this claim.",
        )

    prompt = build_referee_input(
        claim,
        evidence,
        entailments,
        contradiction,
    )
    try:
        content = (request or request_decision)(prompt)
        decision = RefereeDecision.model_validate_json(content)
    except RefereeError:
        raise
    except (ValidationError, ValueError) as error:
        raise RefereeError("Groq returned an invalid referee decision.") from error
    except Exception as error:
        logger.error("Referee request failed with %s.", type(error).__name__)
        raise RefereeError("The referee service is unavailable.") from error

    try:
        cited_ids = [UUID(item) for item in decision.evidence_ids]
    except ValueError as error:
        raise RefereeError("The referee returned an invalid evidence ID.") from error

    available_ids = {item.id for item in evidence}
    if not set(cited_ids).issubset(available_ids):
        raise RefereeError("The referee cited an unknown evidence ID.")

    return Verdict(
        claim_id=claim.id,
        label=decision.label,
        confidence=decision.confidence,
        evidence_ids=cited_ids,
        explanation=decision.explanation,
    )
