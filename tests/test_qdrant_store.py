from typing import cast

import pytest
from httpx import Headers
from pydantic import SecretStr
from qdrant_client import QdrantClient
from qdrant_client import models as qmodels
from qdrant_client.http.exceptions import UnexpectedResponse

from context9.config import Settings
from context9.models import DocumentChunk
from context9.qdrant_store import (
    QdrantAuthenticationError,
    QdrantDocumentStore,
    QdrantStoreError,
    _raise_qdrant_error,
    _secret_value,
    _validate_vectors,
    collection_name,
)


class FakeQueryResponse:
    def __init__(self, points: list[qmodels.ScoredPoint]) -> None:
        """Initialize the fake query response."""
        self.points = points


class FakeQdrantClient:
    def __init__(self) -> None:
        """Initialize the fake Qdrant client."""
        self.collections: set[str] = set()
        self.deleted: list[tuple[str, object]] = []
        self.upserted: list[tuple[str, list[qmodels.PointStruct]]] = []
        self.created: list[tuple[str, qmodels.VectorParams]] = []
        self.query_response = FakeQueryResponse([])
        self.fail_on: str | None = None

    def collection_exists(self, *, collection_name: str) -> bool:
        if self.fail_on == "collection_exists":
            raise _unexpected_response(401)
        return collection_name in self.collections

    def create_collection(self, *, collection_name: str, vectors_config: qmodels.VectorParams) -> None:
        if self.fail_on == "create_collection":
            raise _unexpected_response(401)
        self.collections.add(collection_name)
        self.created.append((collection_name, vectors_config))

    def delete(self, *, collection_name: str, points_selector: object) -> None:
        if self.fail_on == "delete":
            raise _unexpected_response(401)
        self.deleted.append((collection_name, points_selector))

    def upsert(self, *, collection_name: str, points: list[qmodels.PointStruct]) -> None:
        if self.fail_on == "upsert":
            raise _unexpected_response(401)
        self.upserted.append((collection_name, points))

    def query_points(
        self,
        *,
        collection_name: str,
        query: list[float],
        limit: int,
        with_payload: bool,
        with_vectors: bool,
    ) -> FakeQueryResponse:
        assert collection_name
        assert query
        assert limit > 0
        assert with_payload is True
        assert with_vectors is False
        if self.fail_on == "query_points":
            raise _unexpected_response(401)
        return self.query_response


def settings() -> Settings:
    return Settings(
        **cast(
            "dict[str, object]",
            {
                "_env_file": None,
                "qdrant_api_key": SecretStr("secret"),
                "collection_prefix": "Context 9",
                "qdrant_timeout_seconds": 5,
            },
        )
    )


def document_chunk(**overrides: object) -> DocumentChunk:
    payload: dict[str, object] = {
        "id": "chunk-1",
        "package": "FastAPI",
        "version": "latest",
        "source_url": "https://fastapi.tiangolo.com/",
        "title": "Docs",
        "text": "Dependency injection",
        "ordinal": 0,
        "metadata": {"section": "dependencies", "nested": {"ok": True}},
    }
    payload.update(overrides)
    return DocumentChunk.model_validate(payload)


def store_with_fake_client() -> tuple[QdrantDocumentStore, FakeQdrantClient]:
    fake = FakeQdrantClient()
    return QdrantDocumentStore(settings(), client=cast("QdrantClient", fake)), fake


def _unexpected_response(status_code: int) -> UnexpectedResponse:
    return UnexpectedResponse(
        status_code=status_code,
        reason_phrase="Unauthorized" if status_code == 401 else "Internal Server Error",
        content=b"Invalid API key or JWT",
        headers=Headers(),
    )


def _trigger_auth_failure(store: QdrantDocumentStore, fake: FakeQdrantClient, operation: str) -> None:
    if operation == "query_points":
        fake.collections.add("context_9_fastapi_latest")
        store.search(package="fastapi", version="latest", vector=[0.1], limit=1)
        return
    store.upsert_chunks([document_chunk()], [[0.1, 0.2]])


def test_collection_name_sanitizes_parts() -> None:
    assert collection_name("Context 9", "Fast API", "v1.0") == "context_9_fast_api_v1_0"
    assert collection_name("!!!", "@@@", "###") == "context9"


def test_qdrant_store_upserts_chunks_and_payloads() -> None:
    store, fake = store_with_fake_client()
    chunk = document_chunk()

    collection = store.upsert_chunks([chunk], [[0.1, 0.2]])

    assert collection == "context_9_fastapi_latest"
    assert fake.created[0][1].size == 2
    assert fake.deleted
    assert fake.upserted[0][0] == collection
    point = fake.upserted[0][1][0]
    assert point.id == "chunk-1"
    assert point.vector == [0.1, 0.2]
    assert point.payload is not None
    assert point.payload["source_url"] == "https://fastapi.tiangolo.com/"
    assert point.payload["metadata"] == {"section": "dependencies", "nested": {"ok": True}}


@pytest.mark.parametrize(
    ("chunks", "vectors", "message"),
    [
        ([], [], "at least one chunk"),
        ([document_chunk()], [], "same length"),
        ([document_chunk()], [[]], "must not be empty"),
        ([document_chunk(), document_chunk(id="chunk-2", ordinal=1)], [[0.1], [0.1, 0.2]], "same length"),
    ],
)
def test_qdrant_store_rejects_invalid_upserts(
    chunks: list[DocumentChunk],
    vectors: list[list[float]],
    message: str,
) -> None:
    store, _fake = store_with_fake_client()

    with pytest.raises(ValueError, match=message):
        store.upsert_chunks(chunks, vectors)


def test_qdrant_store_search_returns_empty_when_collection_is_missing() -> None:
    store, _fake = store_with_fake_client()

    assert store.search(package="fastapi", version="latest", vector=[0.1], limit=5) == []


def test_qdrant_store_ensure_collection_rejects_invalid_vector_size() -> None:
    store, _fake = store_with_fake_client()

    with pytest.raises(ValueError, match="vector_size"):
        store.ensure_collection("collection", 0)


def test_qdrant_store_ensure_collection_skips_existing_collection() -> None:
    store, fake = store_with_fake_client()
    fake.collections.add("collection")

    store.ensure_collection("collection", 2)

    assert fake.created == []


def test_qdrant_store_search_converts_scored_points() -> None:
    store, fake = store_with_fake_client()
    fake.collections.add("context_9_fastapi_latest")
    fake.query_response = FakeQueryResponse(
        [
            qmodels.ScoredPoint(
                id="chunk-1",
                version=1,
                score=0.9,
                payload={
                    "text": "Dependency injection",
                    "package": "fastapi",
                    "version": "latest",
                    "source_url": "https://fastapi.tiangolo.com/",
                    "title": "Docs",
                    "ordinal": 2,
                    "metadata": {
                        "ok": True,
                        "nested": {"count": 1},
                        "list": ["a", 1, object()],
                        1: "dropped",
                        "bad": object(),
                    },
                },
            )
        ]
    )

    candidates = store.search(package="fastapi", version="latest", vector=[0.1, 0.2], limit=3)

    assert len(candidates) == 1
    assert candidates[0].id == "chunk-1"
    assert candidates[0].metadata == {"ok": True, "nested": {"count": 1}, "list": ["a", 1]}


@pytest.mark.parametrize(
    ("vector", "limit", "message"),
    [
        ([], 1, "query vector"),
        ([0.1], 0, "limit"),
    ],
)
def test_qdrant_store_search_rejects_invalid_inputs(vector: list[float], limit: int, message: str) -> None:
    store, _fake = store_with_fake_client()

    with pytest.raises(ValueError, match=message):
        store.search(package="fastapi", version="latest", vector=vector, limit=limit)


@pytest.mark.parametrize(
    "operation",
    [
        "collection_exists",
        "create_collection",
        "delete",
        "upsert",
        "query_points",
    ],
)
def test_qdrant_store_translates_authentication_errors(operation: str) -> None:
    store, fake = store_with_fake_client()
    fake.fail_on = operation

    with pytest.raises(QdrantAuthenticationError, match="QDRANT_API_KEY"):
        _trigger_auth_failure(store, fake, operation)


def test_qdrant_store_translates_unexpected_errors() -> None:
    with pytest.raises(QdrantStoreError, match="Qdrant request failed"):
        _raise_qdrant_error(_unexpected_response(500))


def test_validate_vectors_rejects_empty_vector_collection() -> None:
    with pytest.raises(ValueError, match="at least one vector"):
        _validate_vectors([])


def test_secret_value_unwraps_secret() -> None:
    assert _secret_value(None) is None
    assert _secret_value(SecretStr("secret")) == "secret"
