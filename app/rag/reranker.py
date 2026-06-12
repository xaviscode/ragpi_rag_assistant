from __future__ import annotations

from typing import Any

from sentence_transformers import CrossEncoder


class Reranker:
    def __init__(self, model_name: str, cache_dir: str):
        self.model = CrossEncoder(model_name, cache_folder=cache_dir)

    def rerank(
        self,
        query: str,
        items: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        if not items:
            return items

        pairs = [(query, item.get("text", "")) for item in items]
        scores = self.model.predict(pairs)

        reranked = []

        for item, score in zip(items, scores):
            copied = dict(item)
            copied["rerank_score"] = float(score)
            reranked.append(copied)

        reranked.sort(key=lambda item: item["rerank_score"], reverse=True)

        return reranked[:top_k]