# Axiom frontend

The Next.js interface for Axiom's evidence-led claim analysis pipeline.

## Local development

Start the FastAPI backend on port 8000, then run:

```bash
cp .env.example .env.local
npm install
npm run dev
```

Open `http://localhost:3000`. The local environment file points the frontend to
`http://127.0.0.1:8000` by default.

## Production

The frontend is deployed on Vercel. Set `NEXT_PUBLIC_API_URL` to the public API
origin before building the production deployment.
