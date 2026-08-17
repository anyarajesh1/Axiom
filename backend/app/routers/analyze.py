import asyncio
import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session

from app.db.models import Claim as ClaimRow
from app.db.models import Submission
from app.db.session import get_session
from app.graph.nodes.extractor import ClaimExtractionError, extract_claims
from app.retrieval.service import retrieve_evidence
from app.retrieval.store import VectorStoreError
from app.retrieval.tavily import ExternalSearchError
from app.schemas import (
    AnalyzeClaimsRequest,
    AnalyzeClaimsResponse,
    AnalyzeEvidenceRequest,
    AnalyzeEvidenceResponse,
)

router = APIRouter(prefix="/analyze", tags=["analysis"])
logger = logging.getLogger(__name__)


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
