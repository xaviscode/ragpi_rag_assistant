# RagPi RAG Assistant (API-only)

A lightweight **RAG (Retrieval-Augmented Generation)** API built with **FastAPI + Chroma**.  
It ingests documents from a folder, chunks and embeds them, stores vectors in Chroma, and answers questions by retrieving relevant chunks and sending them to an LLM via **Ollama**.

---

## What it does

1. **Ingest**
   - Reads files from `/data/raw` (mounted from `./data/raw`)
   - Splits text into chunks (configurable)
   - Creates embeddings (configurable backend/model)
   - Stores embeddings + metadata in **ChromaDB**

2. **Query**
   - Embeds the question
   - Retrieves top-k most relevant chunks from Chroma
   - Builds a prompt with the retrieved context
   - Calls the LLM to generate an answer
   - Optionally returns sources/evidence used

---

## Requirements

- Docker + Docker Compose
- A running LLM endpoint, typically:
  - **Ollama running on the host**
- Internet access only needed to download embedding models on first run.

---

## Project layout

```
.
├── app/                    # FastAPI app + RAG logic
├── data/
│   ├── raw/                # Put PDFs/TXT/MD/etc here
│   ├── chroma/             # Persistent vector DB
│   └── hf_cache/           # Cache for models (HF, etc.)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env
```

---

## Quick start

### 1) Put documents in `./data/raw`

```bash
mkdir -p data/raw
cp your_file.pdf data/raw/
```

Supported types typically include `.pdf`, `.txt`, `.md`, `.html` (depends on the parser installed).

---

### 2) Configure `.env`

Example:

```dotenv
# Protect endpoints (optional)
API_KEY=

# LLM via Ollama on host
LLM_BACKEND=ollama
OLLAMA_BASE_URL=http://host.docker.internal:11434
LLM_MODEL=qwen2.5:7b-instruct
OLLAMA_KEEP_ALIVE=10m
OLLAMA_TIMEOUT=120

# Embeddings
EMBEDDING_BACKEND=fastembed
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5

# Retrieval + limits
TOP_K=3
MAX_NEW_TOKENS=160
MAX_CONTEXT_CHARS=2500
TEMPERATURE=0.0

# Chunking
CHUNK_SIZE_CHARS=1000
CHUNK_OVERLAP_CHARS=150

# Storage
CHROMA_DIR=/data/chroma
HF_HOME=/data/hf_cache
COLLECTION_NAME=docs
ANONYMIZED_TELEMETRY=FALSE
```

Notes:
- `OLLAMA_BASE_URL` must be reachable **from inside the container**.
- `LLM_MODEL` must be present in Ollama (e.g., `ollama pull qwen2.5:7b-instruct`).

---

### 3) Ensure Ollama is running (host)

```bash
sudo systemctl status ollama --no-pager
curl -sS http://127.0.0.1:11434/api/tags | head
```

Pull a model if needed:
```bash
ollama pull qwen2.5:7b-instruct
```

If you do not have ollama see [docs.ollama](https://docs.ollama.com) for details.

---

### 4) Run the API

```bash
docker compose up -d --build
```

API will be at:
- `http://localhost:8000`

---

## Docker compose reference

A typical `docker-compose.yml`:

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
      - ./data/raw:/data/raw
      - ./data/chroma:/data/chroma
      - ./data/hf_cache:/data/hf_cache
    command: ["uvicorn","app.main:app","--host","0.0.0.0","--port","8000","--workers","1"]
    restart: unless-stopped
    extra_hosts:
      - "host.docker.internal:host-gateway"
```

Why `extra_hosts`?
- On Linux, `host.docker.internal` is not always available by default. Mapping it to the Docker gateway makes `http://host.docker.internal:11434` work in most setups.

---

## API endpoints

> Exact paths depend on your router, but the project exposes:

### Ingest documents
Ingest everything under `/data/raw` into Chroma:

```bash
curl -sS -X POST http://127.0.0.1:8000/ingest
```

If `API_KEY` is set:
```bash
curl -sS -X POST http://127.0.0.1:8000/ingest \
  -H "X-API-Key: your_key_here"
```

### Query
Ask a question:

```bash
curl -sS -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question":"What is this document about?","return_sources":true}'
```

Optional flags:
- `return_sources`: return chunk source + id + distance
- `return_evidence`: return short quoted snippets

Example:
```bash
curl -sS -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question":"List the countries mentioned.",
    "return_sources": true,
    "return_evidence": true
  }'
```

### Debug retrieve (retrieval-only)
See what the vector DB retrieves:

```bash
curl -sS -X POST http://127.0.0.1:8000/debug/retrieve \
  -H "Content-Type: application/json" \
  -d '{"question":"Which countries are mentioned?"}'
```

---

## Typical workflow

1. Add/replace files in `./data/raw`
2. Call `POST /ingest`
3. Call `POST /query`

---

## Persistence & reset

Persistent data:
- `./data/chroma` → vector DB
- `./data/hf_cache` → model cache

Rebuild the index from scratch:

```bash
docker compose down
rm -rf data/chroma/*
docker compose up -d --build
curl -sS -X POST http://127.0.0.1:8000/ingest
```

---

## Performance tuning

Key knobs:

- Smaller Ollama model on constrained hardware
- Reduce output length:
  - `MAX_NEW_TOKENS=80–200`
- Reduce context size:
  - `MAX_CONTEXT_CHARS=1500–6000`
- Retrieval depth:
  - `TOP_K=2–8`
- Determinism:
  - `TEMPERATURE=0.0`

Chunking (often good for PDFs):
- `CHUNK_SIZE_CHARS=800–1200`
- `CHUNK_OVERLAP_CHARS=100–200`

---

## Troubleshooting

### “Ollama unreachable … host.docker.internal timed out”
The container can’t reach the Ollama endpoint.

Checks:
```bash
# On host:
curl -sS http://127.0.0.1:11434/api/tags | head

# Inside container:
docker exec -it ragpi python -c "import os,requests; u=os.getenv('OLLAMA_BASE_URL'); print(u); print(requests.get(u+'/api/tags',timeout=5).status_code)"
```

Fixes:
- Keep in compose:
  ```yaml
  extra_hosts:
    - "host.docker.internal:host-gateway"
  ```
- Ensure `.env` matches:
  ```dotenv
  OLLAMA_BASE_URL=http://host.docker.internal:11434
  ```
- Ensure Ollama is running and listening.

### PDF parsing errors (invalid header / EOF marker not found)
Your file may be corrupted or not actually a PDF.

```bash
file data/raw/yourfile.pdf
```

Re-download/re-export the document.

---

## Security notes

If exposed beyond localhost:
- Set `API_KEY` and require `X-API-Key` for ingest/admin endpoints
- Restrict CORS origins
- Consider rate limiting / auth proxy
