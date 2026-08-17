from uuid import uuid4

import pytest

from app.graph.nodes.reranker import RerankerError, rerank_evidence
from app.schemas import Evidence


def make_evidence(text: str) -> Evidence:
    return Evidence(
        id=uuid4(),
        text=text,
        source_name="Test source",
        source_url="https://example.com/source",
        category="science",
        score=0.8,
    )


def test_reranker_orders_by_cross_encoder_score() -> None:
    less_relevant = make_evidence("Mars has a thin atmosphere.")
    more_relevant = make_evidence(
        "Electric vehicle batteries can retain useful capacity after vehicle use."
    )

    results = rerank_evidence(
        "EV batteries may remain useful after leaving a vehicle.",
        [less_relevant, more_relevant],
        predict=lambda _: [-2.0, 3.0],
    )

    assert results[0].id == more_relevant.id
    assert results[0].reranker_score == pytest.approx(0.952574, abs=1e-6)
    assert results[1].score == 0.8


def test_reranker_rejects_wrong_result_count() -> None:
    with pytest.raises(RerankerError, match="unexpected result count"):
        rerank_evidence(
            "A claim",
            [make_evidence("A passage long enough for this test.")],
            predict=lambda _: [],
        )
