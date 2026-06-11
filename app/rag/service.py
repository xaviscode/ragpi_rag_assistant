from __future__ import annotations

import os
import tempfile
from typing import Any
from app.rag.config import Settings
from app.rag.document_store import DocumentStore
from app.rag.embeddings import Embedder
from app.rag.evidence import select_evidence
from app.rag.ingest import ingest_folder
from app.rag.llm import LocalLLM
from app.rag.prompts import build_prompt_parts
from app.rag.retrieve import retrieve
from app.rag.vector_store import VectorStore


class RagService:
    def __init__(self, settings: Settings):
        self.settings = settings
        for d in [settings.hf_home, settings.raw_dir, settings.chroma_dir, settings.metadata_dir]:
            os.makedirs(d, exist_ok=True)
        self.document_store = DocumentStore(settings)
        self.store = VectorStore(settings.chroma_dir, settings.collection_name, settings.anonymized_telemetry)
        self.embedder = Embedder(
            settings.embedding_model,
            settings.hf_home,
            settings.embedding_query_prefix,
            settings.embedding_document_prefix,
        )
        self.llm = LocalLLM(
            settings.llm_model,
            settings.ollama_base_url,
            settings.ollama_timeout,
            settings.ollama_keep_alive,
        )

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "collection_name": self.settings.collection_name,
            "doc_chunks": self.store.count(),
            "documents": self.document_store.count_documents(),
            "max_documents": self.settings.max_documents,
        }

    def list_documents(self) -> dict[str, Any]:
        documents = self.document_store.list_documents()
        return {"documents": documents, "count": len(documents), "max_documents": self.settings.max_documents}

    def ingest_folder(self) -> dict[str, Any]:
        return ingest_folder(self.settings, self.store, self.embedder, self.document_store)

    def _expand_query_with_hyde(self, question: str) -> str:
        if not self.settings.hyde_enabled:
            return question
        prompt = f'''Write a short factual paragraph that would directly answer this question if the information existed in a document.
Do not mention that this is hypothetical. Do not say "I don't know". Use 2 short sentences maximum.

Question:
{question}

Hypothetical answer:'''
        try:
            hypothetical = self.llm.generate_simple(prompt, self.settings.hyde_max_tokens, 0.1).strip()
            if hypothetical:
                return f"{question}\n\n{hypothetical}"
        except Exception as exc:
            print("HyDE query expansion failed:", exc)
        return question

    def upload_documents(self, files: list[Any]) -> dict[str, Any]:
        uploaded, skipped, rejected = [], [], []
        max_bytes = self.settings.max_upload_size_mb * 1024 * 1024
        for file in files:
            filename = file.filename
            if not filename:
                rejected.append({"filename": "unknown", "status": "rejected", "reason": "no_filename"})
                continue
            try:
                content = file.file.read()
                if len(content) > max_bytes:
                    rejected.append({"filename": filename, "status": "rejected", "reason": f"file_too_large_max_{self.settings.max_upload_size_mb}_mb", "bytes": len(content)})
                    continue
                suffix = os.path.splitext(filename)[1].lower()
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(content)
                    tmp_path = tmp.name
                try:
                    doc = self.document_store.register_uploaded_file(tmp_path, filename, getattr(file, "content_type", None))
                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                item = {
                    "document_id": doc.get("document_id"),
                    "filename": doc.get("filename") or filename,
                    "original_filename": doc.get("original_filename") or filename,
                    "bytes": doc.get("file_size_bytes") or len(content),
                    "status": doc.get("upload_status", "uploaded"),
                }
                if doc.get("upload_status") == "duplicate":
                    item["reason"] = "duplicate_content_hash"
                    skipped.append(item)
                else:
                    uploaded.append(item)
            except ValueError as exc:
                rejected.append({"filename": filename, "status": "rejected", "reason": str(exc)})
            except Exception as exc:
                rejected.append({"filename": filename, "status": "rejected", "reason": f"upload_error: {exc}"})
        return {"uploaded": uploaded, "skipped": skipped, "rejected": rejected, "document_count": self.document_store.count_documents(), "max_documents": self.settings.max_documents}

    def delete_document(self, document_id: str) -> dict[str, Any]:
        doc = self.document_store.delete_document_record(document_id)
        if not doc:
            return {"document_id": document_id, "deleted": False, "removed_file": False, "removed_chunks": False}
        removed_file = False
        path = doc.get("path")
        if path and os.path.exists(path):
            os.remove(path)
            removed_file = True
        self.store.delete_by_document_id(document_id)
        return {"document_id": document_id, "deleted": True, "removed_file": removed_file, "removed_chunks": True}

    def debug_retrieve(self, question: str, document_ids: list[str] | None = None) -> dict[str, Any]:
        query_text = self._expand_query_with_hyde(question)
        r = retrieve(self.settings, self.store, self.embedder, question, document_ids, query_text)
        items = [
            {
                "source": it.get("source", "unknown"),
                "document_id": it.get("document_id"),
                "chunk_index": it.get("chunk_index"),
                "distance": it.get("distance"),
                "chunk_id": it.get("chunk_id"),
                "preview": (it.get("text") or "")[:500],
            }
            for it in r["items"]
        ]
        return {
            "question": question,
            "query_text": query_text,
            "top_k": self.settings.top_k,
            "fetch_k": self.settings.fetch_k,
            "distance_threshold": self.settings.distance_threshold,
            "mmr_enabled": self.settings.mmr_enabled,
            "hyde_enabled": self.settings.hyde_enabled,
            "raw_candidates": r.get("raw_candidates"),
            "filtered_candidates": r.get("filtered_candidates"),
            "document_ids": document_ids,
            "items": items,
        }

    def _retry_empty_answer(self, question: str, evidence_text: str) -> str:
        prompt = f'''Use the evidence below to answer the question.
Return a short direct answer.
If the evidence does not answer the question, return exactly:
I don't know based on the provided documents.

Evidence:
{evidence_text}

Question:
{question}

Answer:'''
        try:
            return self.llm.generate_simple(prompt, min(self.settings.max_new_tokens, 180), 0.0).strip()
        except Exception as exc:
            print("LLM retry failed:", exc)
            return ""

    def answer(self, question: str, return_sources: bool = False, return_evidence: bool = True, document_ids: list[str] | None = None) -> dict[str, Any]:
        query_text = self._expand_query_with_hyde(question)
        r = retrieve(self.settings, self.store, self.embedder, question, document_ids, query_text)
        items = r["items"]
        ev = select_evidence(
            question,
            items,
            self.embedder.embed_query,
            self.embedder.embed_texts,
            max_evidence=10,
            max_total_chars=min(self.settings.max_context_chars, 5000),
        )
        evidence_text = (ev.get("evidence_text") or "").strip()
        if not evidence_text:
            out: dict[str, Any] = {"answer": "I don't know based on the provided documents."}
            if return_sources:
                out["sources"] = [{"source": it["source"], "document_id": it.get("document_id"), "chunk_id": it["chunk_id"], "chunk_index": it.get("chunk_index"), "distance": it["distance"]} for it in items]
            if return_evidence:
                out["evidence"] = []
            return out
        system_prompt, user_prompt = build_prompt_parts(evidence_text, question)
        try:
            answer = self.llm.generate(system_prompt, user_prompt, self.settings.max_new_tokens, self.settings.temperature)
        except Exception as exc:
            print("LLM generation failed:", exc)
            answer = ""
        if len(answer.strip()) < self.settings.min_answer_chars:
            answer = self._retry_empty_answer(question, evidence_text)
        if not answer.strip():
            answer = "I don't know based on the provided documents."
        out: dict[str, Any] = {"answer": answer}
        if return_sources:
            out["sources"] = [{"source": it["source"], "document_id": it.get("document_id"), "chunk_id": it["chunk_id"], "chunk_index": it.get("chunk_index"), "distance": it["distance"]} for it in items]
        if return_evidence:
            out["evidence"] = ev["evidence"]
        return out
