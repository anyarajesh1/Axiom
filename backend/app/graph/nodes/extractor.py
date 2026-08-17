import logging
from collections.abc import Callable

from groq import Groq
from pydantic import ValidationError

from app.config import settings
from app.schemas import Claim, ClaimExtraction

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You extract atomic, verifiable factual claims from user text.

Return JSON only with this exact shape:
{"claims": [{"text": "atomic factual claim", "source_span": "exact quote"}]}

Rules:
- Include claims that can be supported or contradicted by evidence.
- Exclude opinions, questions, commands, and purely subjective statements.
- Make each claim understandable on its own.
- Keep source_span as an exact quote from the user's text.
- Split compound and causal statements into their component claims.
- If there are no factual claims, return {"claims": []}.

Examples:
Input: "The Moon orbits Earth, and Mars has two moons."
Output: {"claims": [
  {"text": "The Moon orbits Earth.", "source_span": "The Moon orbits Earth"},
  {"text": "Mars has two moons.", "source_span": "Mars has two moons"}
]}

Input: "EV sales increased because battery prices fell."
Output: {"claims": [
  {"text": "EV sales increased.", "source_span": "EV sales increased"},
  {"text": "Battery prices fell.", "source_span": "battery prices fell"},
  {"text": "Falling battery prices caused EV sales to increase.", "source_span": "EV sales increased because battery prices fell"}
]}

Input: "I love electric cars."
Output: {"claims": []}
"""


class ClaimExtractionError(RuntimeError):
    """Raised when the model response cannot be validated as claims."""


def request_extraction(text: str) -> str:
    with Groq(
        api_key=settings.GROQ_API_KEY,
        timeout=30.0,
        max_retries=2,
    ) as client:
        response = client.chat.completions.create(
            model=settings.GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "claim_extraction",
                    "strict": True,
                    "schema": ClaimExtraction.model_json_schema(),
                },
            },
            temperature=0,
            max_tokens=1200,
        )

    content = response.choices[0].message.content
    if not content:
        raise ClaimExtractionError("Groq returned an empty claim extraction.")
    return content


def extract_claims(
    text: str,
    request: Callable[[str], str] | None = None,
) -> list[Claim]:
    normalized_text = text.strip()
    if not normalized_text:
        raise ValueError("Text must not be empty.")

    extraction_request = request or request_extraction
    try:
        response_content = extraction_request(normalized_text)
    except ClaimExtractionError:
        raise
    except Exception as error:
        logger.error(
            "Claim extraction request failed with %s.",
            type(error).__name__,
        )
        raise ClaimExtractionError(
            "The claim extraction service is unavailable."
        ) from error

    try:
        extraction = ClaimExtraction.model_validate_json(response_content)
    except ValidationError as error:
        raise ClaimExtractionError(
            "Groq returned an invalid claim extraction."
        ) from error

    return [
        Claim(text=item.text, source_span=item.source_span)
        for item in extraction.claims
    ]
