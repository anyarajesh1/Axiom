import asyncio
import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session

from app.db.models import Claim as ClaimRow
from app.db.models import Submission
from app.db.models import Verdict as VerdictRow
from app.db.session import get_session
from app.graph.nodes.entailment import EntailmentError
from app.graph.nodes.extractor import ClaimExtractionError, extract_claims
from app.graph.nodes.reranker import RerankerError
from app.graph.pipeline import run_pipeline
from app.graph.verification import verify_claim
from app.retrieval.service import retrieve_evidence
from app.retrieval.store import VectorStoreError
from app.retrieval.tavily import ExternalSearchError
from app.schemas import (
    AnalyzeClaimsRequest,
    AnalyzeClaimsResponse,
    AnalyzeEvidenceRequest,
    AnalyzeEvidenceResponse,
    AnalyzeRequest,
    AnalyzeResponse,
    VerifyClaimRequest,
    VerifyClaimResponse,
)

router = APIRouter(prefix="/analyze", tags=["analysis"])
logger = logging.getLogger(__name__)


@router.post("", response_model=AnalyzeResponse, status_code=status.HTTP_201_CREATED)
async def analyze(
    payload: AnalyzeRequest,
    session: Annotated[Session, Depends(get_session)],
) -> AnalyzeResponse:
    try:
        pipeline_state = await asyncio.to_thread(run_pipeline, payload.text)
    except ClaimExtractionError as error:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(error),
        ) from error

    submission = Submission(input_text=payload.text)
    claim_rows = [
        ClaimRow(
            id=claim.id,
            submission_id=submission.id,
            text=claim.text,
            source_span=claim.source_span,
        )
        for claim in pipeline_state.claims
    ]
    verdict_rows = [
        VerdictRow(
            claim_id=verdict.claim_id,
            label=verdict.label,
            confidence=verdict.confidence,
            evidence_ids=[str(item) for item in verdict.evidence_ids],
            explanation=verdict.explanation,
        )
        for verdict in pipeline_state.verdicts
    ]

    try:
        session.add(submission)
        session.flush()
        session.add_all(claim_rows)
        session.flush()
        session.add_all(verdict_rows)
        session.commit()
    except Exception as error:
        session.rollback()
        logger.error(
            "Saving the complete analysis failed with %s.",
            type(error).__name__,
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Axiom could not save the completed analysis.",
        ) from error

    return AnalyzeResponse(
        submission_id=submission.id,
        claims=pipeline_state.claims,
        evidence_by_claim=pipeline_state.evidence_by_claim,
        entailment_by_claim=pipeline_state.entailment_by_claim,
        contradictions_by_claim=pipeline_state.contradictions_by_claim,
        verdicts=pipeline_state.verdicts,
    )


@router.post(
    "/claims",
    response_model=AnalyzeClaimsResponse,
    status_code=status.HTTP_201_CREATED,
)
async def analyze_claims(
    payload: AnalyzeClaimsRequest,
    session: Annotated[Session, Depends(get_session)],
) -> AnalyzeClaimsResponse:
    try:
        claims = await asyncio.to_thread(extract_claims, payload.text)
    except ClaimExtractionError as error:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(error),
        ) from error

    submission = Submission(input_text=payload.text)
    claim_rows = [
        ClaimRow(
            id=claim.id,
            submission_id=submission.id,
            text=claim.text,
            source_span=claim.source_span,
        )
        for claim in claims
    ]

    try:
        session.add(submission)
        session.flush()
        session.add_all(claim_rows)
        session.commit()
    except Exception as error:
        session.rollback()
        logger.error(
            "Saving extracted claims failed with %s.",
            type(error).__name__,
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Axiom could not save the extracted claims.",
        ) from error

    return AnalyzeClaimsResponse(
        submission_id=submission.id,
        claims=claims,
    )


@router.post("/evidence", response_model=AnalyzeEvidenceResponse)
async def analyze_evidence(
    payload: AnalyzeEvidenceRequest,
) -> AnalyzeEvidenceResponse:
    try:
        outcome = await asyncio.to_thread(
            retrieve_evidence,
            payload.claim,
            payload.limit,
        )
    except (ExternalSearchError, VectorStoreError) as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(error),
        ) from error

    return AnalyzeEvidenceResponse(
        claim=payload.claim,
        evidence=outcome.evidence,
        external_search_used=outcome.external_search_used,
    )


@router.post("/verify", response_model=VerifyClaimResponse)
async def verify_claim_evidence(
    payload: VerifyClaimRequest,
) -> VerifyClaimResponse:
    try:
        return await asyncio.to_thread(
            verify_claim,
            payload.claim,
            payload.candidate_limit,
            payload.evidence_limit,
        )
    except (
        EntailmentError,
        ExternalSearchError,
        RerankerError,
        VectorStoreError,
    ) as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(error),
        ) from error
