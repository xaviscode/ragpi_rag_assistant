from __future__ import annotations

import os
import inspect
from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self, model_name: str, hf_home: str):
        os.environ["HF_HOME"] = hf_home
        device = os.getenv("DEVICE", "cpu").strip().lower() or "cpu"
        self.model = SentenceTransformer(model_name, device=device)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        vecs = self.model.encode(
            texts,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return vecs.tolist()

    def embed_query(self, text: str) -> list[float]:
        vec = self.model.encode(
            [text],
            show_progress_bar=False,
            normalize_embeddings=True,
        )[0]
        return vec.tolist()