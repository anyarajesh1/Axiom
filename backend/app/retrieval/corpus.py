import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, HttpUrl

from app.config import ROOT_DIR

DEFAULT_CORPUS_PATH = ROOT_DIR / "axiom_corpus" / "seed_passages.jsonl"


class CorpusPassage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=20)
    source_name: str = Field(min_length=2)
    source_url: HttpUrl
    category: str = Field(pattern=r"^[a-z_]+$")


def load_corpus(path: Path = DEFAULT_CORPUS_PATH) -> list[CorpusPassage]:
    """Load and validate a JSON Lines corpus, rejecting duplicate passages."""
    passages: list[CorpusPassage] = []
    seen_text: set[str] = set()

    with path.open(encoding="utf-8") as corpus_file:
        for line_number, line in enumerate(corpus_file, start=1):
            if not line.strip():
                continue

            try:
                passage = CorpusPassage.model_validate_json(line)
            except (json.JSONDecodeError, ValueError) as error:
                raise ValueError(
                    f"Invalid corpus record on line {line_number}."
                ) from error

            normalized_text = " ".join(passage.text.lower().split())
            if normalized_text in seen_text:
                raise ValueError(
                    f"Duplicate corpus passage on line {line_number}."
                )

            seen_text.add(normalized_text)
            passages.append(passage)

    if not passages:
        raise ValueError("The corpus must contain at least one passage.")

    return passages
