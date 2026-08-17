from uuid import uuid4

import pytest
from pytest import MonkeyPatch

from app.graph.nodes import entailment
from app.schemas import Evidence


def make_evidence(text: str) -> Evidence:
    return Evidence(
        id=uuid4(),
        text=text,
        source_name="Test source",
        source_url="https://example.com/source",
        category="science",
        score=1.0,
        reranker_score=0.9,
    )


def test_entailment_maps_model_scores_in_documented_order() -> None:
    contradicting = make_evidence(
        "Many EV batteries retain useful capacity beyond five years."
    )

    results = entailment.score_entailment(
        "Every EV battery must be replaced after five years.",
        [contradicting],
        predict=lambda _: [[0.91, 0.03, 0.06]],
    )

    assert results[0].evidence_id == contradicting.id
    assert results[0].label == "contradiction"
    assert results[0].confidence == 0.91
    assert results[0].scores["entailment"] == 0.03


def test_entailment_uses_evidence_as_premise() -> None:
    evidence = make_evidence("Mars has a thin atmosphere.")
    received_pairs: list[tuple[str, str]] = []

    def predict(pairs: list[tuple[str, str]]) -> list[list[float]]:
        received_pairs.extend(pairs)
        return [[0.01, 0.98, 0.01]]

    entailment.score_entailment(
        "Mars has an atmosphere.",
        [evidence],
        predict=predict,
    )

    assert received_pairs == [
        ("Mars has a thin atmosphere.", "Mars has an atmosphere.")
    ]


def test_hf_fallback_requires_token(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(entailment.settings, "HF_TOKEN", None)

    with pytest.raises(entailment.EntailmentError, match="HF_TOKEN is required"):
        entailment.predict_hf([("Evidence", "Claim")])


def test_hf_output_parser_accepts_named_labels() -> None:
    scores = entailment._parse_hf_scores(
        [
            {"label": "entailment", "score": 0.8},
            {"label": "neutral", "score": 0.15},
            {"label": "contradiction", "score": 0.05},
        ]
    )

    assert scores == [0.05, 0.8, 0.15]
