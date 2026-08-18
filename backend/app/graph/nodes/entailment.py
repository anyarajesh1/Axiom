from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from functools import lru_cache
from typing import TYPE_CHECKING, Any

import httpx
from groq import Groq
from pydantic import BaseModel, ConfigDict, Field, ValidationError

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

GROQ_SYSTEM_PROMPT = """Classify whether each evidence passage entails,
contradicts, or is neutral toward the claim. Use only the supplied text, not
outside knowledge. Return one result for every zero-based evidence index.
Confidence must reflect how directly the passage determines the label.
"""


class RemoteEntailmentItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    index: int = Field(ge=0)
    label: EntailmentLabel
    confidence: float = Field(ge=0, le=1)


class RemoteEntailmentBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    results: list[RemoteEntailmentItem]


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


def request_groq_entailment(prompt: str) -> str:
    with Groq(
        api_key=settings.GROQ_API_KEY,
        timeout=30.0,
        max_retries=2,
    ) as client:
        response = client.chat.completions.create(
            model=settings.GROQ_MODEL,
            messages=[
                {"role": "system", "content": GROQ_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "axiom_entailment_batch",
                    "strict": True,
                    "schema": RemoteEntailmentBatch.model_json_schema(),
                },
            },
            temperature=0,
            max_tokens=1200,
        )
    content = response.choices[0].message.content
    if not content:
        raise EntailmentError("Groq returned an empty NLI response.")
    return content


def predict_groq(
    pairs: list[tuple[str, str]],
    request: Callable[[str], str] | None = None,
) -> Sequence[Sequence[float]]:
    if not pairs:
        return []

    prompt = json.dumps(
        {
            "claim": pairs[0][1],
            "evidence": [
                {"index": index, "text": evidence_text}
                for index, (evidence_text, _) in enumerate(pairs)
            ],
        },
        separators=(",", ":"),
    )
    try:
        content = (request or request_groq_entailment)(prompt)
        batch = RemoteEntailmentBatch.model_validate_json(content)
    except EntailmentError:
        raise
    except (ValidationError, ValueError) as error:
        raise EntailmentError("Groq returned invalid NLI output.") from error
    except Exception as error:
        raise EntailmentError("Axiom could not reach Groq for NLI.") from error

    by_index = {item.index: item for item in batch.results}
    if set(by_index) != set(range(len(pairs))):
        raise EntailmentError("Groq returned an unexpected NLI result set.")

    rows: list[list[float]] = []
    for index in range(len(pairs)):
        result = by_index[index]
        remainder = (1 - result.confidence) / 2
        scores = {label: remainder for label in LABELS}
        scores[result.label] = result.confidence
        rows.append([scores[label] for label in LABELS])
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
        if settings.LOW_MEMORY_MODE:
            inference = predict_groq
        elif settings.USE_HF_INFERENCE_API:
            inference = predict_hf
        else:
            inference = predict_local

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
