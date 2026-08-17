from fastapi.testclient import TestClient
from pytest import MonkeyPatch

import app.main as main_module

client = TestClient(main_module.app)


def test_health_dependencies_are_green(monkeypatch: MonkeyPatch) -> None:
    async def fake_dependency_health() -> dict[str, str]:
        return {"groq": "green", "qdrant": "green", "neon": "green"}

    monkeypatch.setattr(
        main_module,
        "dependency_health",
        fake_dependency_health,
    )

    response = client.get("/health/deps")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "dependencies": {
            "groq": "green",
            "qdrant": "green",
            "neon": "green",
        },
    }
