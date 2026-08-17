from fastapi import FastAPI

from app.config import settings
from app.health import dependency_health

app = FastAPI(title="Axiom API")
app.state.settings = settings


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
