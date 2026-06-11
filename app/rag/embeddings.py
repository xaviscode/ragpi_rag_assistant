from __future__ import annotations

import os
from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self, model_name: str, cache_dir: str, query_prefix: str = "", document_prefix: str = ""):
        os.makedirs(cache_dir, exist_ok=True)
        self.model = SentenceTransformer(model_name, cache_folder=cache_dir)
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        prepared = [f"{self.document_prefix}{t}" for t in texts]
        embeddings = self.model.encode(prepared, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        prepared = f"{self.query_prefix}{text}"
        embeddings = self.model.encode([prepared], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
        return embeddings.tolist()[0]
