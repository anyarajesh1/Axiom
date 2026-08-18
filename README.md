# Axiom

Axiom is an evidence-led claim analysis application. Give it a statement or a
short paragraph and it extracts checkable claims, retrieves relevant sources,
compares supporting and contradictory evidence, and returns an explained
verdict with citations.

## Live application

- Web app: https://axiom-ten-alpha.vercel.app
- API: https://axiom-api-togb.onrender.com
- Interactive API documentation: https://axiom-api-togb.onrender.com/docs

The API runs on a free Render instance and may take up to a minute to wake after
a period of inactivity.

## How it works

1. A Groq-hosted language model extracts individual claims from the input.
2. Axiom retrieves candidate passages from Qdrant and can fall back to Tavily
   web search when the local corpus is not sufficient.
3. The pipeline reranks the candidates and classifies each relationship as
   support, contradiction, or neutral.
4. A referee step weighs the evidence and produces a verdict, confidence score,
   explanation, and source list.
5. Submissions, claims, and verdicts are persisted in Neon Postgres.

The analysis pipeline is orchestrated with LangGraph. Production uses a
low-memory inference path designed for the Render free tier; local development
can use the included sentence-transformer models.

## Technology

- Next.js 16, React 19, TypeScript, and Tailwind CSS
- FastAPI, Pydantic, SQLModel, and LangGraph
- Groq for structured language-model inference
- Qdrant for corpus retrieval
- Tavily for external search fallback
- Neon Postgres for persistence
- Vercel for the frontend and Render for the API
- GitHub Actions for linting, tests, and production builds

## Repository layout

```text
backend/          FastAPI application, analysis graph, tests, and corpus tooling
frontend/         Next.js application
axiom_corpus/     Curated starter passages and source metadata
docs/             Project documentation
.github/workflows Continuous integration
render.yaml       Render backend blueprint
```

## Local development

### 1. Configure the environment

Copy the example file and add your own credentials. Never commit `.env`.

```bash
cp .env.example .env
```

Required service values:

```text
GROQ_API_KEY
QDRANT_URL
QDRANT_API_KEY
DATABASE_URL
TAVILY_API_KEY
```

The defaults in `.env.example` configure the Groq model, local CORS origins,
and the full local inference path.

### 2. Run the backend

Python 3.13 is used by the project.

```bash
cd backend
python3 -m venv .venv
./.venv/bin/python3 -m pip install -r requirements.txt
./.venv/bin/python3 -m uvicorn app.main:app --reload
```

The API will be available at `http://127.0.0.1:8000`. To seed the curated
passages into the configured Qdrant collection, run:

```bash
./.venv/bin/python3 -m scripts.seed_corpus
```

### 3. Run the frontend

In another terminal:

```bash
cd frontend
cp .env.example .env.local
npm install
npm run dev
```

Open `http://localhost:3000`.

## API endpoints

- `GET /health` — API liveness
- `GET /health/deps` — Groq, Qdrant, and Neon connectivity
- `POST /analyze` — complete claim-analysis pipeline
- `POST /analyze/claims` — claim extraction only
- `POST /analyze/evidence` — evidence retrieval only
- `POST /analyze/verify` — retrieval, reranking, and entailment analysis

Example:

```bash
curl -X POST http://127.0.0.1:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"text":"Earthquake magnitude is measured on a logarithmic scale."}'
```

## Quality checks

```bash
cd backend
./.venv/bin/python3 -m ruff check app tests scripts
./.venv/bin/python3 -m pytest -q

cd ../frontend
npm run lint
npm run build
```

## Responsible use

Axiom is a research and demonstration tool, not an infallible fact-checker.
Model outputs and retrieved material can be incomplete or incorrect. Review the
linked sources before relying on a verdict for consequential decisions.

