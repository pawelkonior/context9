import importlib
from typing import TYPE_CHECKING

import pytest

from context9.config import Settings
from context9.fetcher import RawDocument
from context9.models import DocumentChunk, SearchCandidate

if TYPE_CHECKING:
    from collections.abc import Coroutine, Sequence


def make_settings() -> Settings:
    return Settings(embedding_dimensions=8)


class FakeEmbeddingService:
    def __init__(self, dimensions: int) -> None:
        """Initialize the fake embedding service."""
        self.dimensions = dimensions

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _text in texts]


class FakeStore:
    def __init__(self, settings: Settings) -> None:
        """Initialize the fake document store."""
        self.settings = settings
        self.upserted_chunks: Sequence[DocumentChunk] = []
        self.upserted_vectors: Sequence[Sequence[float]] = []

    def upsert_chunks(self, chunks: Sequence[DocumentChunk], vectors: Sequence[Sequence[float]]) -> str:
        self.upserted_chunks = chunks
        self.upserted_vectors = vectors
        return "context9_fastapi_latest"

    def search(
        self,
        *,
        package: str,
        version: str,
        vector: Sequence[float],
        limit: int,
    ) -> list[SearchCandidate]:
        assert package == "fastapi"
        assert version == "latest"
        assert vector == [1.0, 0.0]
        assert limit == 3
        return [
            SearchCandidate.model_validate(
                {
                    "id": "chunk-1",
                    "text": "Hello",
                    "score": 1.0,
                    "package": package,
                    "version": version,
                    "source_url": "https://docs.example.com",
                    "ordinal": 0,
                }
            )
        ]


@pytest.mark.asyncio
async def test_main_runs_fetch_chunk_embed_store_and_search_flow(monkeypatch: pytest.MonkeyPatch) -> None:
    main_module = importlib.import_module("context9.main")

    async def fake_fetch_document(source: str) -> RawDocument:
        assert source == "https://docs.example.com"
        return RawDocument(
            source=source,
            body="<html><head><title>Docs</title></head><body><main><h1>Hello</h1></main></body></html>",
            content_type="text/html",
        )

    monkeypatch.setattr(main_module, "fetch_document", fake_fetch_document)
    monkeypatch.setattr(main_module, "HashEmbeddingService", FakeEmbeddingService)
    monkeypatch.setattr(main_module, "QdrantDocumentStore", FakeStore)
    monkeypatch.setattr(main_module, "get_settings", make_settings)

    result = await main_module.main(
        "https://docs.example.com",
        package="fastapi",
        version="latest",
        search_limit=3,
        query="What is default_response_class in FastAPI",
    )

    assert result.collection == "context9_fastapi_latest"
    assert result.source == "https://docs.example.com"
    assert result.query == "What is default_response_class in FastAPI"
    assert result.title == "Docs"
    assert result.chunks == 1
    assert [candidate.text for candidate in result.candidates] == ["Hello"]


def test_run_prints_flow_result(monkeypatch: pytest.MonkeyPatch) -> None:
    main_module = importlib.import_module("context9.main")
    result = main_module.IngestionFlowResult(
        collection="collection",
        source="https://docs.example.com",
        query="What is default_response_class in FastAPI",
        candidates=[],
        chunks=0,
        title=None,
    )
    printed: list[object] = []

    def fake_run(coroutine: Coroutine[object, object, object]) -> object:
        coroutine.close()
        return result

    monkeypatch.setattr(main_module.asyncio, "run", fake_run)
    monkeypatch.setattr(main_module, "rich_print", printed.append)

    main_module.run()

    assert printed == [result]
