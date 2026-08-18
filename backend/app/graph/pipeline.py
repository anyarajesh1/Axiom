import logging
from typing import TypedDict
from uuid import UUID

from langgraph.graph import END, START, StateGraph

from app.graph.nodes.contradiction_detector import detect_contradictions
from app.graph.nodes.entailment import EntailmentError, score_entailment
from app.graph.nodes.evidence_ranker import rank_evidence
from app.graph.nodes.extractor import extract_claims
from app.graph.nodes.referee import (
    RefereeError,
    decide_verdict,
    fallback_verdict,
)
from app.graph.nodes.reranker import RerankerError, rerank_evidence
from app.retrieval.service import retrieve_evidence
from app.retrieval.store import VectorStoreError
from app.retrieval.tavily import ExternalSearchError
from app.schemas import (
    Claim,
    ContradictionSummary,
    EntailmentResult,
    Evidence,
    PipelineState,
    Verdict,
)

logger = logging.getLogger(__name__)


class AxiomGraphState(TypedDict, total=False):
    input_text: str
    claims: list[Claim]
    evidence_by_claim: dict[UUID, list[Evidence]]
    entailment_by_claim: dict[UUID, list[EntailmentResult]]
    contradictions_by_claim: dict[UUID, ContradictionSummary]
    verdicts: list[Verdict]


def claim_extractor_node(state: AxiomGraphState) -> AxiomGraphState:
    return {"claims": extract_claims(state["input_text"])}


def evidence_retriever_node(state: AxiomGraphState) -> AxiomGraphState:
    evidence_by_claim: dict[UUID, list[Evidence]] = {}
    for claim in state.get("claims", []):
        try:
            outcome = retrieve_evidence(claim.text, limit=8)
            evidence_by_claim[claim.id] = outcome.evidence
        except (ExternalSearchError, VectorStoreError) as error:
            logger.warning(
                "Retrieval failed for claim %s with %s.",
                claim.id,
                type(error).__name__,
            )
            evidence_by_claim[claim.id] = []
    return {"evidence_by_claim": evidence_by_claim}


def reranker_node(state: AxiomGraphState) -> AxiomGraphState:
    evidence_by_claim: dict[UUID, list[Evidence]] = {}
    for claim in state.get("claims", []):
        evidence = state.get("evidence_by_claim", {}).get(claim.id, [])
        try:
            evidence_by_claim[claim.id] = rerank_evidence(
                claim.text,
                evidence,
                top_k=5,
            )
        except RerankerError as error:
            logger.warning("Reranking failed for claim %s: %s", claim.id, error)
            evidence_by_claim[claim.id] = [
                item.model_copy(update={"reranker_score": 0.0})
                for item in evidence[:5]
            ]
    return {"evidence_by_claim": evidence_by_claim}


def entailment_node(state: AxiomGraphState) -> AxiomGraphState:
    entailment_by_claim: dict[UUID, list[EntailmentResult]] = {}
    for claim in state.get("claims", []):
        evidence = state.get("evidence_by_claim", {}).get(claim.id, [])
        try:
            entailment_by_claim[claim.id] = score_entailment(
                claim.text,
                evidence,
            )
        except EntailmentError as error:
            logger.warning(
                "Entailment failed for claim %s: %s",
                claim.id,
                error,
            )
            entailment_by_claim[claim.id] = [
                EntailmentResult(
                    evidence_id=item.id,
                    label="neutral",
                    confidence=0,
                    scores={
                        "contradiction": 0,
                        "entailment": 0,
                        "neutral": 0,
                    },
                )
                for item in evidence
            ]
    return {"entailment_by_claim": entailment_by_claim}


def contradiction_detector_node(state: AxiomGraphState) -> AxiomGraphState:
    summaries = {
        claim.id: detect_contradictions(
            claim.id,
            state.get("evidence_by_claim", {}).get(claim.id, []),
            state.get("entailment_by_claim", {}).get(claim.id, []),
        )
        for claim in state.get("claims", [])
    }
    return {"contradictions_by_claim": summaries}


def evidence_ranker_node(state: AxiomGraphState) -> AxiomGraphState:
    ranked = {
        claim.id: rank_evidence(
            state.get("evidence_by_claim", {}).get(claim.id, []),
            state.get("entailment_by_claim", {}).get(claim.id, []),
        )
        for claim in state.get("claims", [])
    }
    return {"evidence_by_claim": ranked}


def referee_agent_node(state: AxiomGraphState) -> AxiomGraphState:
    verdicts: list[Verdict] = []
    for claim in state.get("claims", []):
        try:
            verdict = decide_verdict(
                claim,
                state.get("evidence_by_claim", {}).get(claim.id, []),
                state.get("entailment_by_claim", {}).get(claim.id, []),
                state["contradictions_by_claim"][claim.id],
            )
        except RefereeError as error:
            logger.warning("Referee failed for claim %s: %s", claim.id, error)
            verdict = fallback_verdict(
                claim,
                "Axiom could not produce a reliable evidence-based verdict.",
            )
        verdicts.append(verdict)
    return {"verdicts": verdicts}


def build_pipeline_graph():
    builder = StateGraph(AxiomGraphState)
    builder.add_node("claim_extractor", claim_extractor_node)
    builder.add_node("evidence_retriever", evidence_retriever_node)
    builder.add_node("reranker", reranker_node)
    builder.add_node("entailment", entailment_node)
    builder.add_node("contradiction_detector", contradiction_detector_node)
    builder.add_node("evidence_ranker", evidence_ranker_node)
    builder.add_node("referee_agent", referee_agent_node)
    builder.add_edge(START, "claim_extractor")
    builder.add_edge("claim_extractor", "evidence_retriever")
    builder.add_edge("evidence_retriever", "reranker")
    builder.add_edge("reranker", "entailment")
    builder.add_edge("entailment", "contradiction_detector")
    builder.add_edge("contradiction_detector", "evidence_ranker")
    builder.add_edge("evidence_ranker", "referee_agent")
    builder.add_edge("referee_agent", END)
    return builder.compile()


graph = build_pipeline_graph()


def run_pipeline(input_text: str) -> PipelineState:
    result = graph.invoke({"input_text": input_text})
    return PipelineState.model_validate(result)
