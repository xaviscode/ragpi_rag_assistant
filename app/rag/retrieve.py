from __future__ import annotations

from typing import Any

from app.rag.config import Settings
from app.rag.embeddings import Embedder
from app.rag.vector_store import VectorStore


def _build_where_filter(document_ids: list[str] | None) -> dict[str, Any] | None:
    if not document_ids:
        return None

    if len(document_ids) == 1:
        return {"document_id": document_ids[0]}

    return {"document_id": {"$in": document_ids}}


def retrieve(
    settings: Settings,
    store: VectorStore,
    embedder: Embedder,
    question: str,
    document_ids: list[str] | None = None,
) -> dict[str, Any]:
    qemb = embedder.embed_query(question)
    where = _build_where_filter(document_ids)

    res = store.query(
        qemb,
        n_results=settings.top_k,
        where=where,
    )

    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    ids = (res.get("ids") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]

    items = []

    for doc, meta, cid, dist in zip(docs, metas, ids, dists):
        meta = meta or {}

        items.append(
            {
                "chunk_id": cid,
                "text": doc,
                "source": meta.get("source", "unknown"),
                "document_id": meta.get("document_id"),
                "chunk_index": meta.get("chunk_index"),
                "distance": float(dist) if dist is not None else None,
            }
        )

    return {"items": items}