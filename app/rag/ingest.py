from __future__ import annotations

import os
import uuid
from typing import Dict

from app.rag.utils import list_files, read_text, clean_text, chunk_text
from app.rag.embeddings import Embedder
from app.rag.vector_store import VectorStore
from app.rag.config import Settings

def ingest_folder(settings: Settings, store: VectorStore, embedder: Embedder) -> Dict[str, int]:
    files = list_files(settings.raw_dir)
    files_processed = 0
    added_chunks = 0

    for path in files:
        files_processed += 1
        try:
            raw = read_text(path)
            text = clean_text(raw)
            if not text:
                continue
            chunks = chunk_text(text, settings.chunk_size_chars, settings.chunk_overlap_chars)
            if not chunks:
                continue

            ids = [str(uuid.uuid4()) for _ in chunks]
            embs = embedder.embed_texts(chunks)
            metadatas = [{"source": os.path.relpath(path, settings.raw_dir)} for _ in chunks]
            store.add(ids=ids, embeddings=embs, documents=chunks, metadatas=metadatas)
            added_chunks += len(chunks)
        except Exception:
            # Skip file on errors (bad pdf, encoding issues, etc.)
            continue

    return {"files_processed": files_processed, "added_chunks": added_chunks}
