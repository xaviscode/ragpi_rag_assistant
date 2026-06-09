from __future__ import annotations

import os
from pydantic import BaseModel


class Settings(BaseModel):
    # Security
    api_key: str | None = os.getenv("API_KEY") or None

    # Models
    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL",
        "sentence-transformers/all-MiniLM-L6-v2",
    )
    llm_model: str = os.getenv("LLM_MODEL", "google/flan-t5-small")

    # Retrieval / chunking
    top_k: int = int(os.getenv("TOP_K", "8"))
    chunk_size_chars: int = int(os.getenv("CHUNK_SIZE_CHARS", "800"))
    chunk_overlap_chars: int = int(os.getenv("CHUNK_OVERLAP_CHARS", "150"))

    # Limits
    max_context_chars: int = int(os.getenv("MAX_CONTEXT_CHARS", "12000"))
    max_new_tokens: int = int(os.getenv("MAX_NEW_TOKENS", "450"))
    temperature: float = float(os.getenv("TEMPERATURE", "0.2"))

    # Document limits
    max_documents: int = int(os.getenv("MAX_DOCUMENTS", "50"))
    max_upload_size_mb: int = int(os.getenv("MAX_UPLOAD_SIZE_MB", "25"))

    # Storage
    chroma_dir: str = os.getenv("CHROMA_DIR", "/data/chroma")
    hf_home: str = os.getenv("HF_HOME", "/data/hf_cache")
    raw_dir: str = os.getenv("RAW_DIR", "/data/raw")
    metadata_dir: str = os.getenv("METADATA_DIR", "/data/metadata")
    documents_registry_path: str = os.getenv(
        "DOCUMENTS_REGISTRY_PATH",
        "/data/metadata/documents.json",
    )
    collection_name: str = os.getenv("COLLECTION_NAME", "docs")

    # Supported input documents
    supported_extensions: tuple[str, ...] = (
        ".txt",
        ".md",
        ".html",
        ".htm",
        ".pdf",
        ".tex",
    )