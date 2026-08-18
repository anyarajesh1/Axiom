from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import lru_cache
from typing import TYPE_CHECKING, Any

import httpx

from app.config import settings
from app.schemas import EntailmentLabel, EntailmentResult, Evidence

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder

LABELS: tuple[EntailmentLabel, ...] = (
    "contradiction",
    "entailment",
    "neutral",
)
HF_INFERENCE_BASE_URL = "https://router.huggingface.co/hf-inference/models"
ScorePairs = Callable[
    [list[tuple[str, str]]],
    Sequence[Sequence[float]],
]


class EntailmentError(RuntimeError):
    """Raised when natural-language inference cannot be completed."""


@lru_cache(maxsize=1)
def get_entailment_model() -> CrossEncoder:
    from sentence_transformers import CrossEncoder

    return CrossEncoder(settings.ENTAILMENT_MODEL)


def predict_local(
    pairs: list[tuple[str, str]],
) -> Sequence[Sequence[float]]:
    return get_entailment_model().predict(
        pairs,
        apply_softmax=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )


def _hf_label(label: str) -> EntailmentLabel:
    normalized = label.lower()
    aliases: dict[str, EntailmentLabel] = {
        "label_0": "contradiction",
        "label_1": "entailment",
        "label_2": "neutral",
    }
    if normalized in aliases:
        return aliases[normalized]
    if normalized in LABELS:
        return normalized  # type: ignore[return-value]
    raise EntailmentError(f"Hugging Face returned unknown label {label!r}.")


def _parse_hf_scores(payload: Any) -> list[float]:
    if isinstance(payload, list) and payload and isinstance(payload[0], list):
        payload = payload[0]
    if not isinstance(payload, list):
        raise EntailmentError("Hugging Face returned invalid NLI output.")

    scores = {label: 0.0 for label in LABELS}
    for item in payload:
        if not isinstance(item, dict):
            raise EntailmentError("Hugging Face returned invalid NLI output.")
        label = _hf_label(str(item.get("label", "")))
        scores[label] = float(item.get("score", 0))
    return [scores[label] for label in LABELS]


def predict_hf(
    pairs: list[tuple[str, str]],
) -> Sequence[Sequence[float]]:
    if not settings.HF_TOKEN:
        raise EntailmentError(
            "HF_TOKEN is required when USE_HF_INFERENCE_API is enabled."
        )

    url = f"{HF_INFERENCE_BASE_URL}/{settings.ENTAILMENT_MODEL}"
    headers = {"Authorization": f"Bearer {settings.HF_TOKEN}"}
    rows: list[list[float]] = []
    try:
        with httpx.Client(timeout=30) as client:
            for evidence_text, claim in pairs:
                response = client.post(
                    url,
                    headers=headers,
                    json={
                        "inputs": {
                            "text": evidence_text,
                            "text_pair": claim,
                        },
                        "parameters": {
                            "function_to_apply": "softmax",
                            "top_k": 3,
                        },
                    },
                )
                response.raise_for_status()
                rows.append(_parse_hf_scores(response.json()))
    except httpx.HTTPError as error:
        raise EntailmentError(
            "Axiom could not reach Hugging Face inference."
        ) from error
    return rows


def score_entailment(
    claim: str,
    evidence: list[Evidence],
    predict: ScorePairs | None = None,
) -> list[EntailmentResult]:
    if not evidence:
        return []

    inference = predict
    if inference is None:
        inference = predict_hf if settings.USE_HF_INFERENCE_API else predict_local

    try:
        rows = list(
            inference([(item.text, claim) for item in evidence])
        )
    except EntailmentError:
        raise
    except Exception as error:
        raise EntailmentError(
            "Axiom could not score evidence entailment."
        ) from error

    if len(rows) != len(evidence):
        raise EntailmentError(
            "The entailment model returned an unexpected result count."
        )

    results: list[EntailmentResult] = []
    for item, row in zip(evidence, rows, strict=True):
        if len(row) != len(LABELS):
            raise EntailmentError(
                "The entailment model returned an unexpected label count."
            )
        scores = {
            label: round(float(score), 6)
            for label, score in zip(LABELS, row, strict=True)
        }
        label = max(LABELS, key=scores.__getitem__)
        results.append(
            EntailmentResult(
                evidence_id=item.id,
                label=label,
                confidence=scores[label],
                scores=scores,
            )
        )
    return results
