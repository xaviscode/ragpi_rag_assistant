from __future__ import annotations

import os
from pydantic import BaseModel


class Settings(BaseModel):
    api_key: str | None = os.getenv("API_KEY") or None

    llm_backend: str = os.getenv("LLM_BACKEND", "ollama")
    llm_model: str = os.getenv("LLM_MODEL", "qwen2.5:7b-instruct")
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
    ollama_keep_alive: str = os.getenv("OLLAMA_KEEP_ALIVE", "10m")
    ollama_timeout: int = int(os.getenv("OLLAMA_TIMEOUT", "120"))

    embedding_backend: str = os.getenv("EMBEDDING_BACKEND", "sentence-transformers")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
    embedding_query_prefix: str = os.getenv("EMBEDDING_QUERY_PREFIX", "")
    embedding_document_prefix: str = os.getenv("EMBEDDING_DOCUMENT_PREFIX", "")

    top_k: int = int(os.getenv("TOP_K", "15"))
    fetch_k: int = int(os.getenv("FETCH_K", "50"))
    max_context_chars: int = int(os.getenv("MAX_CONTEXT_CHARS", "12000"))
    max_new_tokens: int = int(os.getenv("MAX_NEW_TOKENS", "450"))
    temperature: float = float(os.getenv("TEMPERATURE", "0.0"))

    distance_threshold: float | None = (
        float(os.getenv("DISTANCE_THRESHOLD")) if os.getenv("DISTANCE_THRESHOLD") else None
    )
    mmr_enabled: bool = os.getenv("MMR_ENABLED", "TRUE").upper() == "TRUE"
    mmr_lambda: float = float(os.getenv("MMR_LAMBDA", "0.65"))
    hyde_enabled: bool = os.getenv("HYDE_ENABLED", "FALSE").upper() == "TRUE"
    hyde_max_tokens: int = int(os.getenv("HYDE_MAX_TOKENS", "120"))
    min_answer_chars: int = int(os.getenv("MIN_ANSWER_CHARS", "1"))

    chunk_size_chars: int = int(os.getenv("CHUNK_SIZE_CHARS", "800"))
    chunk_overlap_chars: int = int(os.getenv("CHUNK_OVERLAP_CHARS", "150"))
    chunk_source_prefix_enabled: bool = os.getenv("CHUNK_SOURCE_PREFIX_ENABLED", "TRUE").upper() == "TRUE"
    max_evidence_chars: int = int(os.getenv("MAX_EVIDENCE_CHARS", "5000"))
    max_evidence_items: int = int(os.getenv("MAX_EVIDENCE_ITEMS", "10"))

    max_documents: int = int(os.getenv("MAX_DOCUMENTS", "50"))
    max_upload_size_mb: int = int(os.getenv("MAX_UPLOAD_SIZE_MB", "25"))

    raw_dir: str = os.getenv("RAW_DIR", "/data/raw")
    chroma_dir: str = os.getenv("CHROMA_DIR", "/data/chroma")
    metadata_dir: str = os.getenv("METADATA_DIR", "/data/metadata")
    documents_registry_path: str = os.getenv("DOCUMENTS_REGISTRY_PATH", "/data/metadata/documents.json")
    hf_home: str = os.getenv("HF_HOME", "/data/hf_cache")
    collection_name: str = os.getenv("COLLECTION_NAME", "docs")
    anonymized_telemetry: bool = os.getenv("ANONYMIZED_TELEMETRY", "FALSE").upper() == "TRUE"

    supported_extensions: tuple[str, ...] = (".txt", ".md", ".html", ".htm", ".pdf", ".tex")

    reranker_enabled: bool = os.getenv("RERANKER_ENABLED", "FALSE").upper() == "TRUE"
    reranker_model: str = os.getenv(
        "RERANKER_MODEL",
        "cross-encoder/ms-marco-MiniLM-L-6-v2",
    )
    rerank_top_n: int = int(os.getenv("RERANK_TOP_N", "15"))
