# Axiom

### Evidence-led claim analysis with sources you can inspect

[![Next.js](https://img.shields.io/badge/Next.js-16-black?logo=next.js)](https://nextjs.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Python-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Orchestration-1C3C3C)](https://www.langchain.com/langgraph)
[![CI](https://github.com/anyarajesh1/Axiom/actions/workflows/ci.yml/badge.svg)](https://github.com/anyarajesh1/Axiom/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/anyarajesh1/Axiom)](https://github.com/anyarajesh1/Axiom/releases/latest)

> **Status:** Live and deployed — no account required.

## Live demo

**[Try Axiom](https://axiom-ten-alpha.vercel.app)**

- Frontend: [axiom-ten-alpha.vercel.app](https://axiom-ten-alpha.vercel.app)
- Backend API: [axiom-api-togb.onrender.com](https://axiom-api-togb.onrender.com)
- API documentation: [axiom-api-togb.onrender.com/docs](https://axiom-api-togb.onrender.com/docs)

The API uses a free Render instance, so the first analysis after a period of
inactivity may take up to a minute while the service wakes.

---

## What is Axiom?

Axiom turns a statement or short paragraph into an inspectable claim-analysis
report. Instead of returning only an answer, it shows the evidence behind the
result so the user can make the final call.

It can:

- Extract individual, checkable claims from natural-language input
- Retrieve relevant passages from a curated vector corpus
- Search the web when the local corpus is not sufficient
- Separate supporting, contradictory, and neutral material
- Return an explained verdict with confidence and source links
- Persist submissions, claims, and verdicts for later analysis

## How the pipeline works

```text
Input text
    ↓
Claim extraction
    ↓
Qdrant retrieval ── insufficient results ──→ Tavily search
    ↓
Evidence reranking
    ↓
Entailment and contradiction analysis
    ↓
Evidence ranking
    ↓
Referee verdict + explanation + citations
```

LangGraph coordinates the pipeline. Groq provides structured language-model
inference, Qdrant stores the retrieval corpus, Tavily supplies external search
fallbacks, and Neon Postgres stores completed analyses.

## Tech stack

### Frontend

- **Next.js 16** with the App Router
- **React 19** and **TypeScript**
- **Tailwind CSS 4**
- **Vercel** deployment

### Backend and AI

- **FastAPI** and **Pydantic** for the API and schemas
- **LangGraph** for pipeline orchestration
- **Groq** for claim extraction, low-memory inference, and final verdicts
- **Sentence Transformers** for local reranking and entailment
- **SQLModel** for persistence models

### Data and infrastructure

- **Qdrant** for vector and corpus retrieval
- **Tavily** for external search fallback
- **Neon Postgres** for submissions, claims, and verdicts
- **Render** for the production API
- **GitHub Actions** for backend and frontend CI

---

## Getting started locally

### Prerequisites

- Python 3.13
- Node.js 24+
- Credentials for Groq, Qdrant, Neon, and Tavily

### 1. Clone the repository

```bash
git clone https://github.com/anyarajesh1/Axiom.git
cd Axiom
```

### 2. Configure the environment

```bash
cp .env.example .env
```

Add your own values to `.env`:

```text
GROQ_API_KEY
QDRANT_URL
QDRANT_API_KEY
DATABASE_URL
TAVILY_API_KEY
```

Never commit `.env`. It is already excluded by the repository `.gitignore`.

### 3. Start the backend

```bash
cd backend
python3 -m venv .venv
./.venv/bin/python3 -m pip install -r requirements.txt
./.venv/bin/python3 -m uvicorn app.main:app --reload
```

The API will run at `http://127.0.0.1:8000`.

Seed the curated Axiom passages into the configured Qdrant collection:

```bash
./.venv/bin/python3 -m scripts.seed_corpus
```

### 4. Start the frontend

In a second terminal:

```bash
cd frontend
cp .env.example .env.local
npm install
npm run dev
```

Open `http://localhost:3000`.

---

## API

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/health` | API liveness |
| `GET` | `/health/deps` | Groq, Qdrant, and Neon connectivity |
| `POST` | `/analyze` | Complete claim-analysis pipeline |
| `POST` | `/analyze/claims` | Claim extraction only |
| `POST` | `/analyze/evidence` | Evidence retrieval only |
| `POST` | `/analyze/verify` | Retrieval, reranking, and entailment |

Example request:

```bash
curl -X POST http://127.0.0.1:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"text":"Earthquake magnitude is measured on a logarithmic scale."}'
```

## Project structure

```text
Axiom/
├── backend/
│   ├── app/
│   │   ├── db/                 # SQLModel persistence
│   │   ├── graph/              # LangGraph pipeline and analysis nodes
│   │   ├── retrieval/          # Qdrant, Tavily, and source-quality logic
│   │   └── routers/            # FastAPI endpoints
│   ├── scripts/                # Corpus seeding tools
│   └── tests/                  # Backend test suite
├── frontend/
│   └── src/app/                # Next.js interface
├── axiom_corpus/               # Curated passages and source metadata
├── docs/                       # Project documentation
├── .github/workflows/          # Continuous integration
└── render.yaml                 # Render deployment blueprint
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

The current release passes 52 backend tests as well as frontend linting,
TypeScript validation, and a production build.

## Production profile

Local development uses the included sentence-transformer models. The deployed
API uses a low-memory inference path designed for Render's free instance while
preserving the same retrieval, verification, and verdict stages.

## Responsible use

Axiom is a research and demonstration tool, not an infallible fact-checker.
Models and retrieved sources can be incomplete, outdated, or incorrect. Always
review the linked material before using a verdict for an important decision.

---

Built by [Anya Rajesh](https://github.com/anyarajesh1).
