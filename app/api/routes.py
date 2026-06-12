from __future__ import annotations

from fastapi import APIRouter, Depends, File, Header, HTTPException, UploadFile

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


def require_api_key(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> None:
    expected = service.settings.api_key
    if not expected:
        return
    if x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


@router.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(**service.health())


@router.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    return QueryResponse(
        **service.answer(
            req.question,
            req.return_sources,
            req.return_evidence,
            req.document_ids,
        )
    )


@router.post("/ingest", response_model=IngestResponse, dependencies=[Depends(require_api_key)])
def ingest():
    return IngestResponse(**service.ingest_folder())


@router.post("/debug/retrieve", dependencies=[Depends(require_api_key)])
def debug_retrieve(req: QueryRequest):
    return service.debug_retrieve(req.question, req.document_ids)


@router.get("/documents", response_model=DocumentListResponse)
def list_documents():
    return DocumentListResponse(**service.list_documents())


@router.post("/documents", response_model=UploadResponse, dependencies=[Depends(require_api_key)])
def upload_documents(files: list[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    return UploadResponse(**service.upload_documents(files))


@router.delete(
    "/documents/{document_id}",
    response_model=DeleteDocumentResponse,
    dependencies=[Depends(require_api_key)],
)
def delete_document(document_id: str):
    out = service.delete_document(document_id)
    if not out["deleted"]:
        raise HTTPException(status_code=404, detail="Document not found")
    return DeleteDocumentResponse(**out)