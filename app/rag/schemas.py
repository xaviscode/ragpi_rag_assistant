from __future__ import annotations
from pydantic import BaseModel, Field
from typing import List, Optional

class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    return_sources: bool = False
    return_evidence: bool = True

class SourceItem(BaseModel):
    source: str
    chunk_id: str
    distance: float | None = None

class EvidenceItem(BaseModel):
    source: str
    chunk_id: str
    quote: str

class QueryResponse(BaseModel):
    answer: str
    sources: Optional[List[SourceItem]] = None
    evidence: Optional[List[EvidenceItem]] = None

class IngestResponse(BaseModel):
    files_processed: int
    added_chunks: int

class HealthResponse(BaseModel):
    status: str
    collection_name: str
    doc_chunks: int

class UploadResponse(BaseModel):
    filename: str
    bytes: int
    saved_path: str
