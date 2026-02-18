from __future__ import annotations
from typing import Any, Dict

from app.rag.vector_store import VectorStore
from app.rag.embeddings import Embedder
from app.rag.config import Settings

def retrieve(settings: Settings, store: VectorStore, embedder: Embedder, question: str) -> Dict[str, Any]:
    qemb = embedder.embed_query(question)
    res = store.query(qemb, n_results=settings.top_k)

    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    ids = (res.get("ids") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]

    items = []
    for doc, meta, cid, dist in zip(docs, metas, ids, dists):
        items.append({
            "chunk_id": cid,
            "text": doc,
            "source": (meta or {}).get("source", "unknown"),
            "distance": float(dist) if dist is not None else None,
        })
    return {"items": items}
