import pytest

from context9.config import get_settings
from context9.text import build_chunks, chunk_text, extract_html_text, normalize_text


def test_normalize_text_compacts_horizontal_whitespace_and_blank_lines() -> None:
    text = "  First\t line  \n\n\n  Second\r\n line\f  "

    assert normalize_text(text) == "First line\n\nSecond\nline"


def test_extract_html_text_prefers_main_content_and_removes_noise() -> None:
    html = """
    <html>
      <head>
        <title>  API   Reference  </title>
        <script>ignored()</script>
      </head>
      <body>
        <nav>Navigation</nav>
        <main>
          <h1> FastAPI </h1>
          <p> Dependency   injection </p>
          <aside>Related links</aside>
        </main>
        <footer>Footer</footer>
      </body>
    </html>
    """

    title, text = extract_html_text(html)

    assert title == "API Reference"
    assert text == "FastAPI\nDependency injection"


def test_extract_html_text_falls_back_to_article_then_body() -> None:
    article_html = "<html><body><article><h1>Article</h1><p>Body</p></article><p>Ignored</p></body></html>"
    body_html = "<html><body><h1>Body</h1><p>Content</p></body></html>"

    assert extract_html_text(article_html) == (None, "Article\nBody")
    assert extract_html_text(body_html) == (None, "Body\nContent")


def test_extract_html_text_handles_fragment_without_title() -> None:
    assert extract_html_text("<section><p>Only fragment</p></section>") == (None, "Only fragment")


def test_extract_html_text_handles_blank_input() -> None:
    assert extract_html_text("") == (None, "")


def test_extract_html_text_returns_none_for_blank_title() -> None:
    assert extract_html_text("<html><head><title>   </title></head><body>Body</body></html>") == (None, "Body")


def test_chunk_text_groups_paragraphs_without_exceeding_limit() -> None:
    text = "Alpha\n\nBeta\n\nGamma"

    assert chunk_text(text, max_chars=13, overlap=0) == ["Alpha\n\nBeta", "Gamma"]


def test_chunk_text_carries_overlap_between_paragraph_chunks() -> None:
    text = "First paragraph\n\nSecond paragraph\n\nThird"

    assert chunk_text(text, max_chars=33, overlap=5) == ["First paragraph\n\nSecond paragraph", "graph\n\nThird"]


def test_chunk_text_splits_large_block_with_overlap() -> None:
    assert chunk_text("abcdefghij", max_chars=4, overlap=1) == ["abcd", "defg", "ghij"]


def test_chunk_text_flushes_current_before_large_block() -> None:
    assert chunk_text("short\n\nabcdefghijkl", max_chars=5, overlap=1) == ["short", "abcde", "efghi", "ijkl"]


def test_chunk_text_drops_overlap_when_it_would_exceed_limit() -> None:
    assert chunk_text("12345678\n\nabcdefgh", max_chars=10, overlap=5) == ["12345678", "abcdefgh"]


def test_chunk_text_uses_settings_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    get_settings.cache_clear()
    monkeypatch.setenv("CHUNK_MAX_CHARS", "13")
    monkeypatch.setenv("CHUNK_OVERLAP", "0")

    assert chunk_text("Alpha\n\nBeta\n\nGamma") == ["Alpha\n\nBeta", "Gamma"]
    get_settings.cache_clear()


def test_chunk_text_returns_empty_list_for_blank_input() -> None:
    assert chunk_text(" \n\n\t ") == []


@pytest.mark.parametrize(
    ("max_chars", "overlap", "message"),
    [
        (0, 0, "greater than zero"),
        (10, -1, "greater than or equal to zero"),
        (10, 10, "greater than overlap"),
    ],
)
def test_chunk_text_rejects_invalid_options(max_chars: int, overlap: int, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        chunk_text("text", max_chars=max_chars, overlap=overlap)


def test_build_chunks_creates_valid_document_chunks_with_stable_metadata() -> None:
    chunks = build_chunks(
        package="fastapi",
        version="latest",
        source_url="https://fastapi.tiangolo.com/tutorial/dependencies/",
        title="Dependencies",
        text="Alpha\n\nBeta\n\nGamma",
        max_chars=13,
        overlap=0,
    )

    assert [chunk.ordinal for chunk in chunks] == [0, 1]
    assert [chunk.text for chunk in chunks] == ["Alpha\n\nBeta", "Gamma"]
    assert {chunk.package for chunk in chunks} == {"fastapi"}
    assert {chunk.version for chunk in chunks} == {"latest"}
    assert {chunk.title for chunk in chunks} == {"Dependencies"}
    assert all(chunk.metadata["content_sha256"] for chunk in chunks)
    assert (
        chunks[0].id
        == build_chunks(
            package="fastapi",
            version="latest",
            source_url="https://fastapi.tiangolo.com/tutorial/dependencies/",
            title="Dependencies",
            text="Alpha\n\nBeta\n\nGamma",
            max_chars=13,
            overlap=0,
        )[0].id
    )


def test_build_chunks_returns_empty_list_for_blank_text() -> None:
    assert (
        build_chunks(
            package="fastapi",
            version="latest",
            source_url="https://fastapi.tiangolo.com/tutorial/dependencies/",
            title=None,
            text=" ",
        )
        == []
    )
