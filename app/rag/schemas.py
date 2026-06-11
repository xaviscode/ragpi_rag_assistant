from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    return_sources: bool = False
    return_evidence: bool = True
    document_ids: list[str] | None = None


class SourceItem(BaseModel):
    source: str
    chunk_id: str
    distance: float | None = None
    document_id: str | None = None
    chunk_index: int | None = None


class EvidenceItem(BaseModel):
    source: str
    chunk_id: str
    quote: str


class QueryResponse(BaseModel):
    answer: str
    sources: Optional[list[SourceItem]] = None
    evidence: Optional[list[EvidenceItem]] = None


class IngestResponse(BaseModel):
    files_seen: int
    files_processed: int
    files_skipped: int
    files_failed: int
    added_chunks: int
    results: list[dict[str, Any]] = []


class HealthResponse(BaseModel):
    status: str
    collection_name: str
    doc_chunks: int
    documents: int
    max_documents: int


class UploadedDocumentItem(BaseModel):
    document_id: str | None = None
    filename: str
    original_filename: str | None = None
    bytes: int | None = None
    status: str
    reason: str | None = None


class UploadResponse(BaseModel):
    uploaded: list[UploadedDocumentItem] = []
    skipped: list[UploadedDocumentItem] = []
    rejected: list[UploadedDocumentItem] = []
    document_count: int
    max_documents: int


class DocumentItem(BaseModel):
    document_id: str
    filename: str
    original_filename: str
    relative_path: str | None = None
    extension: str | None = None
    file_size_bytes: int | None = None
    status: str
    chunks_count: int = 0
    created_at: str | None = None
    indexed_at: str | None = None
    error: str | None = None


class DocumentListResponse(BaseModel):
    documents: list[DocumentItem]
    count: int
    max_documents: int


class DeleteDocumentResponse(BaseModel):
    document_id: str
    deleted: bool
    removed_file: bool
    removed_chunks: bool
