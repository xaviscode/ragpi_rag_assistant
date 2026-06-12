from app.rag.utils import chunk_text


def test_chunk_text_returns_chunks():
    text = "This is sentence one. This is sentence two. This is sentence three."
    chunks = chunk_text(text, chunk_size=40, overlap=10)

    assert chunks
    assert all(isinstance(chunk, str) for chunk in chunks)
    assert all(chunk.strip() for chunk in chunks)


def test_chunk_text_preserves_short_text():
    text = "AI Research Engineer"
    chunks = chunk_text(text, chunk_size=800, overlap=150)

    assert chunks == ["AI Research Engineer"]


def test_chunk_text_rejects_invalid_overlap():
    try:
        chunk_text("hello", chunk_size=100, overlap=100)
    except ValueError:
        assert True
    else:
        assert False, "Expected ValueError when overlap >= chunk_size"