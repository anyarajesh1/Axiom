from app.retrieval.corpus import load_corpus
from app.retrieval.store import upsert_passages


def main() -> None:
    passages = load_corpus()
    count = upsert_passages(passages)
    print(f"Seeded {count} Axiom passages.")


if __name__ == "__main__":
    main()
