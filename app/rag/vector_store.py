from __future__ import annotations

import os
from typing import Any
import chromadb
from chromadb.config import Settings as ChromaSettings


class VectorStore:
    def __init__(self, persist_dir: str, collection_name: str, anonymized_telemetry: bool = False):
        os.makedirs(persist_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_dir, settings=ChromaSettings(anonymized_telemetry=anonymized_telemetry))
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def count(self) -> int:
        return self.collection.count()

    def add(self, ids: list[str], embeddings: list[list[float]], documents: list[str], metadatas: list[dict[str, Any]]) -> None:
        if ids:
            self.collection.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)

    def query(self, query_embedding: list[float], n_results: int, where: dict[str, Any] | None = None, include_embeddings: bool = False) -> dict[str, Any]:
        include = ["documents", "metadatas", "distances"]
        if include_embeddings:
            include.append("embeddings")
        kwargs: dict[str, Any] = {"query_embeddings": [query_embedding], "n_results": n_results, "include": include}
        if where:
            kwargs["where"] = where
        return self.collection.query(**kwargs)

    def delete_by_document_id(self, document_id: str) -> None:
        self.collection.delete(where={"document_id": document_id})
