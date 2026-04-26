import math

import pytest

from context9.embeddings import HashEmbeddingService, lexical_score, rerank_candidates
from context9.models import SearchCandidate


def search_candidate(*, text: str, score: float) -> SearchCandidate:
    return SearchCandidate.model_validate(
        {
            "id": text.lower().replace(" ", "-"),
            "text": text,
            "score": score,
            "package": "fastapi",
            "version": "latest",
            "source_url": "https://fastapi.tiangolo.com/",
            "ordinal": 0,
        }
    )


def test_hash_embedding_service_returns_deterministic_l2_normalized_vectors() -> None:
    service = HashEmbeddingService(dimensions=32)

    first = service.embed_texts(["Dependency injection"])[0]
    second = service.embed_texts(["Dependency injection"])[0]

    assert first == second
    assert len(first) == 32
    assert math.sqrt(sum(value * value for value in first)) == pytest.approx(1.0)


def test_hash_embedding_service_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="dimensions"):
        HashEmbeddingService(dimensions=0)

    with pytest.raises(ValueError, match="at least one text"):
        HashEmbeddingService(dimensions=8).embed_texts([])


def test_hash_embedding_service_returns_zero_vector_for_blank_text() -> None:
    assert HashEmbeddingService(dimensions=4).embed_texts(["   "]) == [[0.0, 0.0, 0.0, 0.0]]


def test_lexical_score_measures_query_term_overlap() -> None:
    assert lexical_score("fastapi dependency injection", "Dependency injection docs") == pytest.approx(2 / 3)
    assert lexical_score("", "Dependency injection docs") == 0.0
    assert lexical_score("fastapi", "") == 0.0


def test_rerank_candidates_blends_vector_and_lexical_scores() -> None:
    weak_vector_strong_text = search_candidate(text="FastAPI dependency injection", score=0.2)
    strong_vector_weak_text = search_candidate(text="Unrelated", score=0.9)

    ranked = rerank_candidates("dependency injection", [strong_vector_weak_text, weak_vector_strong_text])

    assert [candidate.text for candidate in ranked] == ["Unrelated", "FastAPI dependency injection"]
    assert ranked[0].score == 0.648
    assert ranked[1].score == 0.424
