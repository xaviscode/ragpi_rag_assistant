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

        os.makedirs(self.settings.hf_home, exist_ok=True)
        os.makedirs(self.settings.raw_dir, exist_ok=True)
        os.makedirs(self.settings.chroma_dir, exist_ok=True)
        os.makedirs(self.settings.metadata_dir, exist_ok=True)

        self.document_store = DocumentStore(self.settings)
        self.store = VectorStore(
            self.settings.chroma_dir,
            self.settings.collection_name,
        )
        self.embedder = Embedder(
            self.settings.embedding_model,
            self.settings.hf_home,
        )
        self.llm = LocalLLM(
            self.settings.llm_model,
            self.settings.hf_home,
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

        return {
            "documents": documents,
            "count": len(documents),
            "max_documents": self.settings.max_documents,
        }

    def ingest_folder(self) -> dict[str, Any]:
        return ingest_folder(
            self.settings,
            self.store,
            self.embedder,
            self.document_store,
        )

    def upload_documents(self, files: list[Any]) -> dict[str, Any]:
        uploaded = []
        skipped = []
        rejected = []

        max_bytes = self.settings.max_upload_size_mb * 1024 * 1024

        for file in files:
            filename = file.filename

            if not filename:
                rejected.append(
                    {
                        "filename": "unknown",
                        "status": "rejected",
                        "reason": "no_filename",
                    }
                )
                continue

            try:
                content = file.file.read()

                if len(content) > max_bytes:
                    rejected.append(
                        {
                            "filename": filename,
                            "status": "rejected",
                            "reason": f"file_too_large_max_{self.settings.max_upload_size_mb}_mb",
                            "bytes": len(content),
                        }
                    )
                    continue

                suffix = os.path.splitext(filename)[1].lower()

                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(content)
                    tmp_path = tmp.name

                try:
                    doc = self.document_store.register_uploaded_file(
                        source_path=tmp_path,
                        original_filename=filename,
                        content_type=getattr(file, "content_type", None),
                    )
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
                rejected.append(
                    {
                        "filename": filename,
                        "status": "rejected",
                        "reason": str(exc),
                    }
                )
            except Exception as exc:
                rejected.append(
                    {
                        "filename": filename,
                        "status": "rejected",
                        "reason": f"upload_error: {exc}",
                    }
                )

        return {
            "uploaded": uploaded,
            "skipped": skipped,
            "rejected": rejected,
            "document_count": self.document_store.count_documents(),
            "max_documents": self.settings.max_documents,
        }

    def delete_document(self, document_id: str) -> dict[str, Any]:
        doc = self.document_store.delete_document_record(document_id)

        if not doc:
            return {
                "document_id": document_id,
                "deleted": False,
                "removed_file": False,
                "removed_chunks": False,
            }

        removed_file = False

        path = doc.get("path")
        if path and os.path.exists(path):
            os.remove(path)
            removed_file = True

        self.store.delete_by_document_id(document_id)

        return {
            "document_id": document_id,
            "deleted": True,
            "removed_file": removed_file,
            "removed_chunks": True,
        }

    def debug_retrieve(
        self,
        question: str,
        document_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        r = retrieve(
            self.settings,
            self.store,
            self.embedder,
            question,
            document_ids=document_ids,
        )

        items = []

        for it in r["items"]:
            items.append(
                {
                    "source": it.get("source", "unknown"),
                    "document_id": it.get("document_id"),
                    "chunk_index": it.get("chunk_index"),
                    "distance": it.get("distance"),
                    "chunk_id": it.get("chunk_id"),
                    "preview": (it.get("text") or "")[:400],
                }
            )

        return {
            "question": question,
            "top_k": self.settings.top_k,
            "document_ids": document_ids,
            "items": items,
        }

    def answer(
        self,
        question: str,
        return_sources: bool = False,
        return_evidence: bool = True,
        document_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        r = retrieve(
            self.settings,
            self.store,
            self.embedder,
            question,
            document_ids=document_ids,
        )

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
            out: dict[str, Any] = {
                "answer": "I don't know based on the provided documents."
            }

            if return_sources:
                out["sources"] = [
                    {
                        "source": it["source"],
                        "document_id": it.get("document_id"),
                        "chunk_id": it["chunk_id"],
                        "chunk_index": it.get("chunk_index"),
                        "distance": it["distance"],
                    }
                    for it in items
                ]

            if return_evidence:
                out["evidence"] = []

            return out

        question_prompt, ctx = build_prompt_parts(
            evidence_text=evidence_text,
            question=question,
        )

        answer = self.llm.generate(
            question_prompt,
            ctx,
            max_new_tokens=self.settings.max_new_tokens,
            temperature=self.settings.temperature,
        )

        out = {"answer": answer}

        if return_sources:
            out["sources"] = [
                {
                    "source": it["source"],
                    "document_id": it.get("document_id"),
                    "chunk_id": it["chunk_id"],
                    "chunk_index": it.get("chunk_index"),
                    "distance": it["distance"],
                }
                for it in items
            ]

        if return_evidence:
            out["evidence"] = ev["evidence"]

        return out