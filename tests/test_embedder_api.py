from __future__ import annotations

from typing import TYPE_CHECKING, cast

from fastapi.testclient import TestClient
from pydantic import SecretStr

from context9.config import Settings, get_settings
from context9.embedder_api import create_app, get_embedding_service, run

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pytest


def make_settings(**overrides: object) -> Settings:
    payload = {"_env_file": None}
    payload.update(overrides)
    return Settings(**cast("dict[str, object]", payload))


class FakeEmbeddingService:
    dimensions = 2

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _text in texts]


def test_embedder_api_healthz() -> None:
    client = TestClient(create_app())

    assert client.get("/healthz").json() == {"status": "ok"}


def test_get_embedding_service_uses_configured_dimensions() -> None:
    service = get_embedding_service(make_settings(embedding_dimensions=16))

    assert service.dimensions == 16


def test_embedder_api_embeds_texts_with_dependency_injected_service() -> None:
    app = create_app()
    app.dependency_overrides[get_settings] = lambda: make_settings()
    app.dependency_overrides[get_embedding_service] = FakeEmbeddingService
    client = TestClient(app)

    response = client.post("/embed", json={"texts": ["hello", "world"]})

    assert response.status_code == 200
    assert response.json() == {"dimensions": 2, "vectors": [[1.0, 0.0], [1.0, 0.0]]}


def test_embedder_api_requires_configured_api_key() -> None:
    app = create_app()
    app.dependency_overrides[get_settings] = lambda: make_settings(embedder_api_key=SecretStr("secret"))
    app.dependency_overrides[get_embedding_service] = FakeEmbeddingService
    client = TestClient(app)

    assert client.post("/embed", json={"texts": ["hello"]}).status_code == 401
    assert client.post("/embed", json={"texts": ["hello"]}, headers={"api-key": "wrong"}).status_code == 401
    assert client.post("/embed", json={"texts": ["hello"]}, headers={"api-key": "secret"}).status_code == 200


def test_embedder_api_reranks_candidates() -> None:
    app = create_app()
    app.dependency_overrides[get_settings] = lambda: make_settings()
    client = TestClient(app)
    payload = {
        "query": "dependency injection",
        "candidates": [
            {
                "id": "a",
                "text": "Unrelated",
                "score": 0.9,
                "package": "fastapi",
                "version": "latest",
                "source_url": "https://fastapi.tiangolo.com/",
                "ordinal": 0,
            },
            {
                "id": "b",
                "text": "Dependency injection",
                "score": 0.2,
                "package": "fastapi",
                "version": "latest",
                "source_url": "https://fastapi.tiangolo.com/",
                "ordinal": 1,
            },
        ],
    }

    response = client.post("/rerank", json=payload)

    assert response.status_code == 200
    assert [candidate["id"] for candidate in response.json()["results"]] == ["a", "b"]


def test_run_starts_uvicorn(monkeypatch: pytest.MonkeyPatch) -> None:
    called: dict[str, object] = {}
    host = "0.0.0.0"  # noqa: S104

    def fake_run(app_path: str, *, host: str, port: int) -> None:
        called.update({"app_path": app_path, "host": host, "port": port})

    monkeypatch.setattr("context9.embedder_api.uvicorn.run", fake_run)

    run()

    assert called == {"app_path": "context9.embedder_api:app", "host": host, "port": 8500}
