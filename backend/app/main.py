import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.db.session import create_db_and_tables
from app.graph.verification import warm_verification_models
from app.health import dependency_health
from app.routers.analyze import router as analyze_router


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    await asyncio.to_thread(create_db_and_tables)
    await asyncio.to_thread(warm_verification_models)
    yield


app = FastAPI(title="Axiom API", lifespan=lifespan)
app.state.settings = settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.frontend_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)
app.include_router(analyze_router)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/health/deps")
async def health_dependencies() -> dict[str, object]:
    dependencies = await dependency_health()
    status = (
        "ok"
        if all(value == "green" for value in dependencies.values())
        else "degraded"
    )
    return {"status": status, "dependencies": dependencies}
