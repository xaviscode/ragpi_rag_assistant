from __future__ import annotations

import os
from typing import Any, Dict

from app.rag.config import Settings
from app.rag.vector_store import VectorStore
from app.rag.embeddings import Embedder
from app.rag.llm import LocalLLM
from app.rag.ingest import ingest_folder
from app.rag.retrieve import retrieve
from app.rag.prompts import build_prompt_parts
from app.rag.evidence import select_evidence

class RagService:
    def __init__(self, settings: Settings):
        self.settings = settings
        os.makedirs(self.settings.hf_home, exist_ok=True)
        os.makedirs(self.settings.raw_dir, exist_ok=True)
        os.makedirs(self.settings.chroma_dir, exist_ok=True)

        self.store = VectorStore(self.settings.chroma_dir, self.settings.collection_name)
        self.embedder = Embedder(self.settings.embedding_model, self.settings.hf_home)
        self.llm = LocalLLM(self.settings.llm_model, self.settings.hf_home)

    def health(self) -> Dict[str, Any]:
        return {
            "status": "ok",
            "collection_name": self.settings.collection_name,
            "doc_chunks": self.store.count(),
        }

    def ingest_folder(self) -> Dict[str, int]:
        return ingest_folder(self.settings, self.store, self.embedder)
    
    def debug_retrieve(self, question: str) -> Dict[str, Any]:
        r = retrieve(self.settings, self.store, self.embedder, question)
        # return top chunks with text preview
        items = []
        for it in r["items"]:
            items.append({
                "source": it.get("source", "unknown"),
                "distance": it.get("distance"),
                "chunk_id": it.get("chunk_id"),
                "preview": (it.get("text") or "")[:400],
            })
        return {"question": question, "top_k": self.settings.top_k, "items": items}

    def answer(self, question: str, return_sources: bool = False, return_evidence: bool = True) -> Dict[str, Any]:
        r = retrieve(self.settings, self.store, self.embedder, question)
        items = r["items"]

        ev = select_evidence(
            question=question,
            retrieved_items=items,
            embed_query_fn=self.embedder.embed_query,
            embed_texts_fn=self.embedder.embed_texts,
            max_evidence=10,
            max_total_chars=min(self.settings.max_context_chars, 3800),
        )
        evidence_text = (ev.get("evidence_text") or "").strip()

        if not evidence_text:
            out: Dict[str, Any] = {"answer": "I don't know based on the provided documents."}
            if return_sources:
                out["sources"] = [{"source": it["source"], "chunk_id": it["chunk_id"], "distance": it["distance"]} for it in items]
            if return_evidence:
                out["evidence"] = []
            return out

        question_prompt, ctx = build_prompt_parts(evidence_text=evidence_text, question=question)
        answer = self.llm.generate(
            question_prompt, ctx,
            max_new_tokens=self.settings.max_new_tokens,
            temperature=self.settings.temperature,
        )

        out: Dict[str, Any] = {"answer": answer}
        if return_sources:
            out["sources"] = [{"source": it["source"], "chunk_id": it["chunk_id"], "distance": it["distance"]} for it in items]
        if return_evidence:
            out["evidence"] = ev["evidence"]
        return out
