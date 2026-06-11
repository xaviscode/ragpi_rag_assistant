from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.rag.config import Settings


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: str) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def safe_filename(filename: str) -> str:
    return Path(filename).name


class DocumentStore:
    def __init__(self, settings: Settings):
        self.settings = settings
        os.makedirs(self.settings.raw_dir, exist_ok=True)
        os.makedirs(self.settings.metadata_dir, exist_ok=True)
        self.path = self.settings.documents_registry_path
        if not os.path.exists(self.path):
            self._write({"documents": []})

    def _read(self) -> dict[str, Any]:
        if not os.path.exists(self.path):
            return {"documents": []}
        with open(self.path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _write(self, data: dict[str, Any]) -> None:
        tmp_path = f"{self.path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, self.path)

    def list_documents(self) -> list[dict[str, Any]]:
        return self._read().get("documents", [])

    def count_documents(self) -> int:
        return len(self.list_documents())

    def get_document(self, document_id: str) -> dict[str, Any] | None:
        return next((d for d in self.list_documents() if d.get("document_id") == document_id), None)

    def get_by_hash(self, content_hash: str) -> dict[str, Any] | None:
        return next((d for d in self.list_documents() if d.get("content_hash") == content_hash), None)

    def get_by_path(self, path: str) -> dict[str, Any] | None:
        abs_path = os.path.abspath(path)
        return next((d for d in self.list_documents() if os.path.abspath(d.get("path", "")) == abs_path), None)

    def can_add_documents(self, n: int = 1) -> bool:
        return self.count_documents() + n <= self.settings.max_documents

    def register_uploaded_file(self, source_path: str, original_filename: str, content_type: str | None = None) -> dict[str, Any]:
        if not os.path.exists(source_path):
            raise FileNotFoundError(source_path)
        ext = Path(original_filename).suffix.lower()
        if ext not in self.settings.supported_extensions:
            raise ValueError(f"Unsupported file extension: {ext}")
        content_hash = sha256_file(source_path)
        existing = self.get_by_hash(content_hash)
        if existing:
            return {**existing, "upload_status": "duplicate"}
        if not self.can_add_documents(1):
            raise ValueError(f"Document limit reached: max_documents={self.settings.max_documents}")
        document_id = str(uuid.uuid4())
        clean_name = safe_filename(original_filename)
        stored_filename = f"{document_id}{ext}"
        dest_path = os.path.join(self.settings.raw_dir, stored_filename)
        shutil.copyfile(source_path, dest_path)
        doc = {
            "document_id": document_id,
            "filename": stored_filename,
            "original_filename": clean_name,
            "path": dest_path,
            "relative_path": os.path.relpath(dest_path, self.settings.raw_dir),
            "extension": ext,
            "content_hash": content_hash,
            "file_size_bytes": os.path.getsize(dest_path),
            "content_type": content_type,
            "status": "uploaded",
            "chunks_count": 0,
            "created_at": utc_now_iso(),
            "indexed_at": None,
            "error": None,
        }
        data = self._read()
        data.setdefault("documents", []).append(doc)
        self._write(data)
        return {**doc, "upload_status": "uploaded"}

    def register_existing_file(self, path: str) -> dict[str, Any]:
        ext = Path(path).suffix.lower()
        if ext not in self.settings.supported_extensions:
            raise ValueError(f"Unsupported file extension: {ext}")
        content_hash = sha256_file(path)
        existing = self.get_by_hash(content_hash)
        if existing:
            return existing
        if not self.can_add_documents(1):
            raise ValueError(f"Document limit reached: max_documents={self.settings.max_documents}")
        clean_name = safe_filename(os.path.basename(path))
        document_id = str(uuid.uuid4())
        doc = {
            "document_id": document_id,
            "filename": clean_name,
            "original_filename": clean_name,
            "path": path,
            "relative_path": os.path.relpath(path, self.settings.raw_dir),
            "extension": ext,
            "content_hash": content_hash,
            "file_size_bytes": os.path.getsize(path),
            "content_type": None,
            "status": "uploaded",
            "chunks_count": 0,
            "created_at": utc_now_iso(),
            "indexed_at": None,
            "error": None,
        }
        data = self._read()
        data.setdefault("documents", []).append(doc)
        self._write(data)
        return doc

    def mark_indexed(self, document_id: str, chunks_count: int) -> None:
        data = self._read()
        for doc in data.get("documents", []):
            if doc.get("document_id") == document_id:
                doc["status"] = "indexed"
                doc["chunks_count"] = chunks_count
                doc["indexed_at"] = utc_now_iso()
                doc["error"] = None
                break
        self._write(data)

    def mark_failed(self, document_id: str, error: str) -> None:
        data = self._read()
        for doc in data.get("documents", []):
            if doc.get("document_id") == document_id:
                doc["status"] = "failed"
                doc["error"] = error
                break
        self._write(data)

    def delete_document_record(self, document_id: str) -> dict[str, Any] | None:
        data = self._read()
        removed = None
        remaining = []
        for doc in data.get("documents", []):
            if doc.get("document_id") == document_id:
                removed = doc
            else:
                remaining.append(doc)
        data["documents"] = remaining
        self._write(data)
        return removed
