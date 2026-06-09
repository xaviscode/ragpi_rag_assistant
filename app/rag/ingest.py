from __future__ import annotations

import os
from typing import Any

from app.rag.config import Settings
from app.rag.document_store import DocumentStore
from app.rag.embeddings import Embedder
from app.rag.utils import chunk_text, clean_text, list_files, read_text
from app.rag.vector_store import VectorStore


def _chunk_id(document_id: str, chunk_index: int) -> str:
    return f"{document_id}::chunk::{chunk_index}"


def ingest_folder(
    settings: Settings,
    store: VectorStore,
    embedder: Embedder,
    document_store: DocumentStore,
) -> dict[str, Any]:
    files = list_files(settings.raw_dir)

    files_seen = 0
    files_processed = 0
    files_skipped = 0
    files_failed = 0
    added_chunks = 0

    results: list[dict[str, Any]] = []

    for path in files:
        files_seen += 1

        try:
            ext = os.path.splitext(path)[1].lower()
            if ext not in settings.supported_extensions:
                files_skipped += 1
                results.append(
                    {
                        "path": path,
                        "status": "skipped",
                        "reason": "unsupported_extension",
                    }
                )
                continue

            doc = document_store.get_by_path(path)

            if doc is None:
                doc = document_store.register_existing_file(path)

            document_id = doc["document_id"]

            if doc.get("status") == "indexed" and doc.get("chunks_count", 0) > 0:
                files_skipped += 1
                results.append(
                    {
                        "document_id": document_id,
                        "source": doc.get("original_filename"),
                        "status": "skipped",
                        "reason": "already_indexed",
                        "chunks_count": doc.get("chunks_count", 0),
                    }
                )
                continue

            raw = read_text(path)
            text = clean_text(raw)

            if not text:
                document_store.mark_failed(document_id, "empty_text")
                files_failed += 1
                results.append(
                    {
                        "document_id": document_id,
                        "source": doc.get("original_filename"),
                        "status": "failed",
                        "reason": "empty_text",
                    }
                )
                continue

            chunks = chunk_text(
                text,
                settings.chunk_size_chars,
                settings.chunk_overlap_chars,
            )

            if not chunks:
                document_store.mark_failed(document_id, "no_chunks_created")
                files_failed += 1
                results.append(
                    {
                        "document_id": document_id,
                        "source": doc.get("original_filename"),
                        "status": "failed",
                        "reason": "no_chunks_created",
                    }
                )
                continue

            # Safety: remove previous chunks for this document before re-indexing.
            store.delete_by_document_id(document_id)

            ids = [_chunk_id(document_id, i) for i in range(len(chunks))]
            embeddings = embedder.embed_texts(chunks)

            metadatas = [
                {
                    "document_id": document_id,
                    "source": doc.get("original_filename") or doc.get("filename"),
                    "relative_path": doc.get("relative_path"),
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "content_hash": doc.get("content_hash"),
                    "extension": doc.get("extension"),
                }
                for i in range(len(chunks))
            ]

            store.add(
                ids=ids,
                embeddings=embeddings,
                documents=chunks,
                metadatas=metadatas,
            )

            document_store.mark_indexed(document_id, len(chunks))

            files_processed += 1
            added_chunks += len(chunks)

            results.append(
                {
                    "document_id": document_id,
                    "source": doc.get("original_filename"),
                    "status": "indexed",
                    "chunks_count": len(chunks),
                }
            )

        except Exception as exc:
            files_failed += 1

            doc = document_store.get_by_path(path)
            if doc:
                document_store.mark_failed(doc["document_id"], str(exc))

            results.append(
                {
                    "path": path,
                    "status": "failed",
                    "reason": str(exc),
                }
            )

    return {
        "files_seen": files_seen,
        "files_processed": files_processed,
        "files_skipped": files_skipped,
        "files_failed": files_failed,
        "added_chunks": added_chunks,
        "results": results,
    }