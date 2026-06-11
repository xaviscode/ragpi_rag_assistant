# RagPi RAG Assistant

A local **Retrieval-Augmented Generation (RAG)** assistant built with **FastAPI**, **ChromaDB**, **SentenceTransformers**, and **Ollama**.

The project lets you place or upload documents, ingest them into a local vector database, and ask questions that are answered using only the indexed document evidence.

The goal is to provide a lightweight local API for document-grounded question answering, with source tracking, evidence extraction, and configurable retrieval behavior.

---

## What it does

RagPi provides a local API that can:

- Register and manage a local document collection.
- Ingest documents from `data/raw`.
- Split documents into sentence-aware chunks.
- Embed chunks with a configurable SentenceTransformers model.
- Store chunks and metadata in ChromaDB.
- Retrieve relevant chunks for a user question.
- Optionally diversify retrieved chunks with MMR-style selection.
- Optionally use HyDE query expansion.
- Generate grounded answers through a local Ollama model.
- Return sources and evidence used to answer.
- Refuse unsupported answers when evidence is insufficient.

---

## Main features

### Document management

- Supports up to 50 managed documents by default.
- Stores raw files locally under `data/raw`.
- Stores a document registry in `data/metadata/documents.json`.
- Tracks document metadata:
  - `document_id`
  - `filename`
  - `original_filename`
  - `content_hash`
  - `status`
  - `chunks_count`
  - `created_at`
  - `indexed_at`
  - `error`
- Detects duplicate documents using SHA-256 hashes.
- Supports document upload, listing, ingestion, deletion, and filtered querying.

### Ingestion

- Reads files from `/data/raw`.
- Supports `.pdf`, `.txt`, `.md`, `.html`, and `.htm`.
- Uses sentence-aware chunking instead of blind character splitting.
- Adds an optional source prefix to chunks before embedding.
- Creates deterministic chunk IDs: `document_id::chunk::chunk_index`.
- Stores chunk-level metadata in Chroma.
- Ingestion is idempotent: running `/ingest` multiple times does not duplicate chunks.

### Retrieval

The project includes improved retrieval behavior:

- Configurable `TOP_K`.
- Configurable `FETCH_K`.
- Optional distance threshold filtering.
- MMR-style candidate diversification.
- Optional HyDE query expansion.
- Source and evidence return.
- Document-specific filtering using `document_ids`.

### Generation

- Uses Ollama as the local LLM backend.
- Answers only from retrieved evidence.
- Returns `I don't know based on the provided documents.` when evidence is insufficient.
- Includes an empty-answer guard and retry prompt to avoid blank model responses.

---

## Project layout

```text
.
├── app/
│   ├── main.py
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py
│   └── rag/
│       ├── __init__.py
│       ├── config.py
│       ├── document_store.py
│       ├── embeddings.py
│       ├── evidence.py
│       ├── ingest.py
│       ├── llm.py
│       ├── prompts.py
│       ├── retrieve.py
│       ├── schemas.py
│       ├── service.py
│       ├── utils.py
│       └── vector_store.py
├── data/
│   ├── raw/
│   ├── chroma/
│   ├── metadata/
│   └── hf_cache/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## Architecture

```text
User
  ↓
FastAPI API
  ↓
Document store
  ↓
Ingestion pipeline
  ↓
Sentence-aware chunking
  ↓
SentenceTransformers embeddings
  ↓
ChromaDB vector store
  ↓
Retriever
  ↓
Evidence selector
  ↓
Prompt builder
  ↓
Ollama LLM
  ↓
Answer + sources + evidence
```

---

## Requirements

- Docker
- Docker Compose
- Ollama running on the host machine
- A local Ollama model, for example:

```bash
ollama pull qwen2.5:7b-instruct
```

Internet access is only needed when downloading models for the first time.

---

## Quick start

### 1. Create local data folders

```bash
mkdir -p data/raw data/chroma data/metadata data/hf_cache
```

### 2. Create `.env`

```bash
cp .env.example .env
```

### 3. Start Ollama

Make sure Ollama is running locally:

```bash
curl http://localhost:11434/api/tags
```

Test the model:

```bash
ollama run qwen2.5:7b-instruct "Say hello in one sentence."
```

### 4. Start the API

```bash
docker compose up -d --build
```

The API runs at:

```text
http://localhost:8000
```

### 5. Check health

```bash
curl http://localhost:8000/health
```

Example response:

```json
{
  "status": "ok",
  "collection_name": "docs",
  "doc_chunks": 0,
  "documents": 0,
  "max_documents": 50
}
```

---

## Docker Compose

The project uses a single service:

```yaml
services:
  ragpi:
    build: .
    container_name: ragpi
    ports:
      - "8000:8000"
    env_file:
      - .env
    volumes:
      - ./data:/data
    command: ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
    restart: unless-stopped
    extra_hosts:
      - "host.docker.internal:host-gateway"
```

The volume mount:

```yaml
volumes:
  - ./data:/data
```

maps local project storage to the container:

```text
Container path        Local path
/data/raw       ->    ./data/raw
/data/chroma    ->    ./data/chroma
/data/metadata  ->    ./data/metadata
/data/hf_cache  ->    ./data/hf_cache
```

This makes documents, Chroma data, metadata, and model cache persistent across container restarts.

---

## Environment variables

Example `.env`:

```dotenv
# Protect admin endpoints such as ingest, upload and delete.
# Leave empty for local development.
API_KEY=

# LLM through Ollama running on the host
LLM_BACKEND=ollama
OLLAMA_BASE_URL=http://host.docker.internal:11434
LLM_MODEL=qwen2.5:7b-instruct
OLLAMA_KEEP_ALIVE=10m
OLLAMA_TIMEOUT=120

# Embeddings
EMBEDDING_BACKEND=sentence-transformers
EMBEDDING_MODEL=BAAI/bge-base-en-v1.5
EMBEDDING_QUERY_PREFIX=
EMBEDDING_DOCUMENT_PREFIX=

# Retrieval and generation
TOP_K=15
FETCH_K=50
MAX_NEW_TOKENS=450
MAX_CONTEXT_CHARS=12000
TEMPERATURE=0.0

# Retrieval quality controls
DISTANCE_THRESHOLD=
MMR_ENABLED=TRUE
MMR_LAMBDA=0.65
HYDE_ENABLED=FALSE
HYDE_MAX_TOKENS=120
MIN_ANSWER_CHARS=1

# Chunking
CHUNK_SIZE_CHARS=800
CHUNK_OVERLAP_CHARS=150
CHUNK_SOURCE_PREFIX_ENABLED=TRUE

# Document limits
MAX_DOCUMENTS=50
MAX_UPLOAD_SIZE_MB=25

# Storage inside the container
RAW_DIR=/data/raw
CHROMA_DIR=/data/chroma
METADATA_DIR=/data/metadata
DOCUMENTS_REGISTRY_PATH=/data/metadata/documents.json
HF_HOME=/data/hf_cache
COLLECTION_NAME=docs
ANONYMIZED_TELEMETRY=FALSE
```

---

## Important configuration notes

### Embedding model

The default improved embedding model is:

```dotenv
EMBEDDING_MODEL=BAAI/bge-base-en-v1.5
```

After changing the embedding model, rebuild the vector index because existing Chroma vectors were created with the previous embedding model.

### Retrieval depth

```dotenv
TOP_K=15
FETCH_K=50
```

`FETCH_K` retrieves more candidates first. The system can then filter and diversify candidates before selecting the final `TOP_K`.

### Distance threshold

```dotenv
DISTANCE_THRESHOLD=
```

By default this is empty, meaning no hard threshold is applied.

You can tune it later after inspecting `/debug/retrieve` distances. A threshold that is too strict may remove useful chunks.

### MMR-style diversification

```dotenv
MMR_ENABLED=TRUE
MMR_LAMBDA=0.65
```

This reduces repeated or overly similar chunks in the final context.

### HyDE query expansion

```dotenv
HYDE_ENABLED=FALSE
```

HyDE is optional. When enabled, the LLM generates a short hypothetical answer to improve the retrieval query. It can help semantic retrieval, but it adds one extra LLM call per query.

Recommended initial setting:

```dotenv
HYDE_ENABLED=FALSE
```

Enable it only after baseline retrieval has been tested.

---

## Add documents

### Option A: add files manually

Place files in:

```text
data/raw
```

Example:

```bash
cp your_file.pdf data/raw/
```

Then ingest:

```bash
curl -X POST http://localhost:8000/ingest
```

### Option B: upload files through the API

```bash
curl -X POST "http://localhost:8000/documents" -F "files=@your_file.pdf"
```

If `API_KEY` is set:

```bash
curl -X POST "http://localhost:8000/documents" -H "X-API-Key: your_key_here" -F "files=@your_file.pdf"
```

---

## Ingest documents

```bash
curl -X POST http://localhost:8000/ingest
```

If `API_KEY` is set:

```bash
curl -X POST http://localhost:8000/ingest -H "X-API-Key: your_key_here"
```

Example first ingestion response:

```json
{
  "files_seen": 6,
  "files_processed": 6,
  "files_skipped": 0,
  "files_failed": 0,
  "added_chunks": 98,
  "results": []
}
```

Example second ingestion response:

```json
{
  "files_seen": 6,
  "files_processed": 0,
  "files_skipped": 6,
  "files_failed": 0,
  "added_chunks": 0,
  "results": []
}
```

The second response confirms idempotency.

---

## List documents

```bash
curl http://localhost:8000/documents
```

Example response:

```json
{
  "documents": [
    {
      "document_id": "uuid",
      "filename": "document.pdf",
      "original_filename": "document.pdf",
      "relative_path": "document.pdf",
      "extension": ".pdf",
      "file_size_bytes": 123456,
      "status": "indexed",
      "chunks_count": 12,
      "created_at": "...",
      "indexed_at": "...",
      "error": null
    }
  ],
  "count": 1,
  "max_documents": 50
}
```

---

## Query documents

Ask a question over all indexed documents:

```bash
curl -X POST http://localhost:8000/query -H "Content-Type: application/json" -d '{
    "question": "What is this document about?",
    "return_sources": true,
    "return_evidence": true
  }'
```

Response:

```json
{
  "answer": "...",
  "sources": [
    {
      "source": "document.pdf",
      "chunk_id": "document_id::chunk::0",
      "distance": 0.52,
      "document_id": "document_id",
      "chunk_index": 0
    }
  ],
  "evidence": [
    {
      "source": "document.pdf",
      "chunk_id": "document_id::chunk::0",
      "quote": "Relevant evidence quote..."
    }
  ]
}
```

---

## Query selected documents

Use `document_ids` to restrict retrieval to specific documents:

```bash
curl -X POST http://localhost:8000/query -H "Content-Type: application/json" -d '{
    "question": "What does this document say about the job position?",
    "document_ids": ["your-document-id"],
    "return_sources": true,
    "return_evidence": true
  }'
```

---

## Debug retrieval

Use `/debug/retrieve` to inspect retrieval before LLM generation:

```bash
curl -X POST http://localhost:8000/debug/retrieve -H "Content-Type: application/json" -d '{
    "question": "To what job position have I applied at Nexthink?"
  }'
```

The response includes:

```text
query_text
top_k
fetch_k
distance_threshold
mmr_enabled
hyde_enabled
raw_candidates
filtered_candidates
items
```

This endpoint is used to make a diagnosing whether failures come from retrieval or generation.

---

## Delete a document

```bash
curl -X DELETE http://localhost:8000/documents/your-document-id
```

If `API_KEY` is set:

```bash
curl -X DELETE http://localhost:8000/documents/your-document-id -H "X-API-Key: your_key_here"
```

Deletion removes:

```text
- The document registry entry.
- The physical file from data/raw.
- The document chunks from Chroma.
```

---

## Reset and re-index

When changing any of these settings:

```text
EMBEDDING_MODEL
CHUNK_SIZE_CHARS
CHUNK_OVERLAP_CHARS
CHUNK_SOURCE_PREFIX_ENABLED
```

rebuild the index:

```bash
docker compose down
rm -rf data/chroma data/metadata
mkdir -p data/chroma data/metadata
docker compose up -d --build
curl -X POST http://localhost:8000/ingest
```

This keeps raw files but recreates the document registry and Chroma vectors.

---

## Troubleshooting

### `/debug/retrieve` works but `/query` fails

This usually means retrieval is working but Ollama generation failed.

Check Ollama:

```bash
curl http://localhost:11434/api/tags
```

Test the model:

```bash
ollama run qwen2.5:7b-instruct "Say hello."
```

### Docker cannot reach Ollama

Make sure `docker-compose.yml` includes:

```yaml
extra_hosts:
  - "host.docker.internal:host-gateway"
```

and `.env` contains:

```dotenv
OLLAMA_BASE_URL=http://host.docker.internal:11434
```

### Empty answers

The current code includes a guard for empty LLM responses. If Ollama returns blank text, the system retries with a simpler prompt and then falls back to:

```text
I don't know based on the provided documents.
```

### Irrelevant chunks are retrieved

Use `/debug/retrieve` to inspect distances and retrieved chunks.

Possible tuning options:

```dotenv
TOP_K=10
FETCH_K=40
MMR_ENABLED=TRUE
DISTANCE_THRESHOLD=
HYDE_ENABLED=FALSE
```

Do not set a strict `DISTANCE_THRESHOLD` until you inspect actual retrieval distances.

---

## Current limitations

This project is optimized for local document-grounded Q&A, but some tasks remain harder than others.

Works well for:

```text
- Semantic questions
- Role extraction from cover letters
- Document-specific facts
- Evidence-backed answers
- Document filtering by ID
- Collection-wide aggregation
- Dates
```

Can be weaker for:

```text
- Exact phone numbers
- Emails
- IDs
- URLs
```

These exact-value tasks may require future work.

---

## Security notes

For local development, `API_KEY` can be empty.

If exposing the API beyond localhost:

- Set `API_KEY`.
- Require `X-API-Key` for upload, ingestion and deletion.
- Restrict CORS origins.
- Add rate limiting or an authentication proxy.
- Avoid exposing raw personal documents publicly.

---

## Credits

This project was created as a practical local RAG assistant for working with personal and academic documents. It was especially useful for exploring large, dense, and technical files by making it easier to ask grounded questions, retrieve relevant evidence, and study complex material more efficiently.

Beyond being a software project, RagPi served as a hands-on learning experience in document ingestion, vector search, local LLMs, retrieval quality, evidence grounding, and API-based system design.
