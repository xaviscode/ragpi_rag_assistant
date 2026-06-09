from __future__ import annotations

from fastapi import APIRouter, File, Header, HTTPException, UploadFile

from app.rag.config import Settings
from app.rag.schemas import (
    DeleteDocumentResponse,
    DocumentListResponse,
    HealthResponse,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    UploadResponse,
)
from app.rag.service import RagService


router = APIRouter()

settings = Settings()
service = RagService(settings)


def _check_api_key(x_api_key: str | None) -> None:
    if settings.api_key:
        if not x_api_key or x_api_key != settings.api_key:
            raise HTTPException(status_code=401, detail="Missing/invalid X-API-Key")


@router.get("/health", response_model=HealthResponse)
def health():
    info = service.health()
    return HealthResponse(**info)


@router.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    out = service.answer(
        req.question,
        return_sources=req.return_sources,
        return_evidence=req.return_evidence,
        document_ids=req.document_ids,
    )
    return QueryResponse(**out)


@router.post("/ingest", response_model=IngestResponse)
def ingest(x_api_key: str | None = Header(default=None, alias="X-API-Key")):
    _check_api_key(x_api_key)
    out = service.ingest_folder()
    return IngestResponse(**out)


@router.post("/debug/retrieve")
def debug_retrieve(req: QueryRequest):
    return service.debug_retrieve(
        req.question,
        document_ids=req.document_ids,
    )


@router.get("/documents", response_model=DocumentListResponse)
def list_documents():
    out = service.list_documents()
    return DocumentListResponse(**out)


@router.post("/documents", response_model=UploadResponse)
def upload_documents(
    files: list[UploadFile] = File(...),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
):
    _check_api_key(x_api_key)

    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    out = service.upload_documents(files)
    return UploadResponse(**out)


@router.delete("/documents/{document_id}", response_model=DeleteDocumentResponse)
def delete_document(
    document_id: str,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
):
    _check_api_key(x_api_key)

    out = service.delete_document(document_id)

    if not out["deleted"]:
        raise HTTPException(status_code=404, detail="Document not found")

    return DeleteDocumentResponse(**out)