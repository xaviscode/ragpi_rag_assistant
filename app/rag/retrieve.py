from __future__ import annotations

from typing import Any
import math
from app.rag.config import Settings
from app.rag.embeddings import Embedder
from app.rag.vector_store import VectorStore


def _build_where_filter(document_ids: list[str] | None) -> dict[str, Any] | None:
    if not document_ids:
        return None
    if len(document_ids) == 1:
        return {"document_id": document_ids[0]}
    return {"document_id": {"$in": document_ids}}


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _mmr_select(candidates: list[dict[str, Any]], query_embedding: list[float], k: int, lambda_mult: float) -> list[dict[str, Any]]:
    if len(candidates) <= k:
        return candidates
    remaining = candidates[:]
    remaining.sort(key=lambda item: item.get("distance", float("inf")))
    selected = [remaining.pop(0)]
    while remaining and len(selected) < k:
        best_idx, best_score = 0, float("-inf")
        for idx, candidate in enumerate(remaining):
            c_emb = candidate.get("embedding")
            if c_emb is None:
                relevance, diversity_penalty = 1.0 - float(candidate.get("distance", 1.0)), 0.0
            else:
                relevance = _cosine(query_embedding, c_emb)
                diversity_penalty = max((_cosine(c_emb, s["embedding"]) for s in selected if s.get("embedding") is not None), default=0.0)
            score = lambda_mult * relevance - (1.0 - lambda_mult) * diversity_penalty
            if score > best_score:
                best_score, best_idx = score, idx
        selected.append(remaining.pop(best_idx))
    return selected


def retrieve(
    settings: Settings,
    store: VectorStore,
    embedder: Embedder,
    question: str,
    document_ids: list[str] | None = None,
    query_text: str | None = None,
    reranker: Any | None = None,
) -> dict[str, Any]:
    effective_query = query_text or question
    qemb = embedder.embed_query(effective_query)
    where = _build_where_filter(document_ids)

    n_results = max(settings.top_k, settings.fetch_k)

    include_embeddings = settings.mmr_enabled
    res = store.query(
        query_embedding=qemb,
        n_results=n_results,
        where=where,
        include_embeddings=include_embeddings,
    )

    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    ids = (res.get("ids") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]
    embs = (res.get("embeddings") or [[]])[0] if include_embeddings else [None] * len(docs)

    candidates = []

    for doc, meta, cid, dist, emb in zip(docs, metas, ids, dists, embs):
        meta = meta or {}

        if settings.distance_threshold is not None and dist is not None:
            if float(dist) > settings.distance_threshold:
                continue

        candidates.append(
            {
                "chunk_id": cid,
                "text": doc,
                "source": meta.get("source", "unknown"),
                "document_id": meta.get("document_id"),
                "chunk_index": meta.get("chunk_index"),
                "distance": float(dist) if dist is not None else None,
                "embedding": emb,
            }
        )

    ranked_candidates = candidates

    if reranker is not None and ranked_candidates:
        ranked_candidates = reranker.rerank(
            query=question,
            items=ranked_candidates,
            top_k=min(settings.rerank_top_n, len(ranked_candidates)),
        )

    if settings.mmr_enabled:
        items = _mmr_select(
            ranked_candidates,
            qemb,
            settings.top_k,
            settings.mmr_lambda,
        )
    else:
        items = ranked_candidates[: settings.top_k]

    for item in items:
        item.pop("embedding", None)

    return {
        "items": items,
        "query_text": effective_query,
        "raw_candidates": len(docs),
        "filtered_candidates": len(candidates),
        "reranked_candidates": len(ranked_candidates),
    }