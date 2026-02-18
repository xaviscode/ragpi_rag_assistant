from __future__ import annotations

import os
from fastapi import APIRouter, UploadFile, File, HTTPException, Header
from app.rag.service import RagService
from app.rag.config import Settings
from app.rag.schemas import QueryRequest, QueryResponse, IngestResponse, HealthResponse, UploadResponse

router = APIRouter()
settings = Settings()
service = RagService(settings)

def _check_api_key(x_api_key: str | None):
    if settings.api_key:
        if not x_api_key or x_api_key != settings.api_key:
            raise HTTPException(status_code=401, detail="Missing/invalid X-API-Key")

@router.get("/health", response_model=HealthResponse)
def health():
    info = service.health()
    return HealthResponse(**info)

@router.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    out = service.answer(req.question, return_sources=req.return_sources, return_evidence=req.return_evidence)
    return QueryResponse(**out)

@router.post("/ingest", response_model=IngestResponse)
def ingest(x_api_key: str | None = Header(default=None, alias="X-API-Key")):
    _check_api_key(x_api_key)
    out = service.ingest_folder()
    return IngestResponse(**out)

@router.post("/debug/retrieve")
def debug_retrieve(req: QueryRequest):
    return service.debug_retrieve(req.question)

@router.post("/documents", response_model=UploadResponse)
def upload_document(
    file: UploadFile = File(...),
    x_api_key: str | None = Header(default=None, alias="X-API-Key")
):
    _check_api_key(x_api_key)
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    raw_dir = settings.raw_dir
    os.makedirs(raw_dir, exist_ok=True)
    dest_path = os.path.join(raw_dir, file.filename)

    if os.path.abspath(dest_path).startswith(os.path.abspath(raw_dir)) is False:
        raise HTTPException(status_code=400, detail="Invalid filename/path")

    content = file.file.read()
    with open(dest_path, "wb") as f:
        f.write(content)

    return UploadResponse(filename=file.filename, bytes=len(content), saved_path=dest_path)
