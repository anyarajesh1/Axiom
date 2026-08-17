from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from app.config import settings
from app.graph.nodes.entailment import (
    get_entailment_model,
    score_entailment,
)
from app.graph.nodes.reranker import get_reranker, rerank_evidence
from app.retrieval.service import retrieve_evidence
from app.schemas import EntailmentResult, Evidence, VerifyClaimResponse


class VerificationState(TypedDict, total=False):
    claim: str
    candidate_limit: int
    evidence_limit: int
    evidence: list[Evidence]
    entailments: list[EntailmentResult]
    external_search_used: bool


def retrieve_node(state: VerificationState) -> VerificationState:
    outcome = retrieve_evidence(
        state["claim"],
        limit=state.get("candidate_limit", 8),
    )
    return {
        "evidence": outcome.evidence,
        "external_search_used": outcome.external_search_used,
    }


def rerank_node(state: VerificationState) -> VerificationState:
    evidence = rerank_evidence(
        state["claim"],
        state.get("evidence", []),
        top_k=state.get("evidence_limit", 5),
    )
    return {"evidence": evidence}


def entailment_node(state: VerificationState) -> VerificationState:
    entailments = score_entailment(
        state["claim"],
        state.get("evidence", []),
    )
    return {"entailments": entailments}


def build_verification_graph():
    builder = StateGraph(VerificationState)
    builder.add_node("retriever", retrieve_node)
    builder.add_node("reranker", rerank_node)
    builder.add_node("entailment", entailment_node)
    builder.add_edge(START, "retriever")
    builder.add_edge("retriever", "reranker")
    builder.add_edge("reranker", "entailment")
    builder.add_edge("entailment", END)
    return builder.compile()


verification_graph = build_verification_graph()


def verify_claim(
    claim: str,
    candidate_limit: int = 8,
    evidence_limit: int = 5,
) -> VerifyClaimResponse:
    result = verification_graph.invoke(
        {
            "claim": claim,
            "candidate_limit": candidate_limit,
            "evidence_limit": evidence_limit,
        }
    )
    return VerifyClaimResponse(
        claim=claim,
        evidence=result.get("evidence", []),
        entailments=result.get("entailments", []),
        external_search_used=result.get("external_search_used", False),
    )


def warm_verification_models() -> None:
    """Load each configured local cross-encoder once during app startup."""
    get_reranker()
    if not settings.USE_HF_INFERENCE_API:
        get_entailment_model()
