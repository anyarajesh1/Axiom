import pytest

from app.graph.nodes.extractor import ClaimExtractionError, extract_claims


def test_extract_claims_returns_typed_atomic_claims() -> None:
    response = """{
      "claims": [
        {
          "text": "EV sales increased.",
          "source_span": "EV sales increased"
        },
        {
          "text": "Battery prices fell.",
          "source_span": "battery prices fell"
        },
        {
          "text": "Falling battery prices caused EV sales to increase.",
          "source_span": "EV sales increased because battery prices fell"
        }
      ]
    }"""

    claims = extract_claims(
        "EV sales increased because battery prices fell.",
        request=lambda _: response,
    )

    assert len(claims) == 3
    assert claims[2].text == (
        "Falling battery prices caused EV sales to increase."
    )


def test_extract_claims_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        extract_claims("   ", request=lambda _: "{}")


def test_extract_claims_rejects_invalid_model_output() -> None:
    with pytest.raises(ClaimExtractionError):
        extract_claims(
            "The Moon orbits Earth.",
            request=lambda _: "not valid JSON",
        )
