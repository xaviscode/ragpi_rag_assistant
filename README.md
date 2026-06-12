# RagPi RAG Assistant

A local **Retrieval-Augmented Generation (RAG)** assistant built with **FastAPI**, **ChromaDB**, **SentenceTransformers**, and **Ollama**.

RagPi lets you index local documents, ask questions over them, and receive answers grounded only in retrieved document evidence. It is designed as a lightweight local API for document-grounded question answering, with source tracking, evidence extraction, document management, configurable retrieval, and local LLM generation.

---

## Overview

RagPi provides a local RAG pipeline that can ingest documents, split them into sentence-aware chunks, embed them with a SentenceTransformers model, store them in ChromaDB, retrieve relevant evidence for a question, and generate a grounded answer through Ollama.

The project focuses on reproducibility, transparency and practical local use. It is especially useful for exploring personal, academic and technical documents without relying on external hosted LLM APIs.

---

## Main features

RagPi supports document upload, local-folder ingestion, document listing, deletion, filtered querying by document ID, source return, evidence return and grounded refusal when evidence is insufficient.

The ingestion pipeline supports `.pdf`, `.txt`, `.md`, `.html`, `.htm` and `.tex` files. It detects duplicate documents using SHA-256 hashes and also detects edited files during re-ingestion. When a file changes, RagPi updates its metadata, removes stale Chroma chunks and re-indexes the new content.

Retrieval is configurable through `TOP_K`, `FETCH_K`, optional distance filtering, MMR-style diversification, optional HyDE query expansion and optional cross-encoder reranking. The default embedding model is `BAAI/bge-base-en-v1.5`, with the recommended BGE query prefix supported through configuration.

Generation uses Ollama locally. The Ollama integration uses the chat API with separate system and user messages. For Qwen3-style models, thinking mode is disabled during normal RAG answers to avoid empty responses caused by the model spending the full token budget on internal reasoning.

---

## Architecture

```text
User
  ↓
FastAPI API
  ↓
Document registry
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
Optional MMR / reranking
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
│       ├── reranker.py
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
├── eval/
│   ├── EVALUATION.md
│   ├── questions.json
│   ├── run_eval.py
│   └── results/
├── tests/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── pytest.ini
├── .env.example
├── .gitignore
└── README.md
```

---

## Requirements

You need Docker, Docker Compose and Ollama running on the host machine.

Pull a local Ollama model before starting the API:

```bash
ollama pull qwen2.5:7b-instruct
```

You can also use Qwen3 models. RagPi disables thinking mode for normal RAG answers when using the Ollama chat API.

Internet access is only required when downloading models or Python packages for the first time.

---

## Quick start

Create the local data folders:

```bash
mkdir -p data/raw data/chroma data/metadata data/hf_cache
```

Create your local environment file:

```bash
cp .env.example .env
```

Make sure Ollama is running:

```bash
curl http://localhost:11434/api/tags
```

Start the API:

```bash
docker compose up -d --build
```

Check health:

```bash
curl http://localhost:8000/health
```

The API runs at:

```text
http://localhost:8000
```

---

## Add and ingest documents

Place files in `data/raw`:

```bash
cp your_file.pdf data/raw/
```

Then ingest:

```bash
curl -X POST http://localhost:8000/ingest
```

If `API_KEY` is configured:

```bash
curl -X POST http://localhost:8000/ingest \
  -H "X-API-Key: your_key_here"
```

You can also upload through the API:

```bash
curl -X POST http://localhost:8000/documents \
  -F "files=@your_file.pdf"
```

Supported formats:

```text
.pdf, .txt, .md, .html, .htm, .tex
```

Ingestion is idempotent. Running it repeatedly does not duplicate chunks. If a previously indexed file changes, RagPi detects the new hash, removes stale chunks and re-indexes the updated file.

---

## Query documents

Ask a question over all indexed documents:

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is this document about?",
    "return_sources": true,
    "return_evidence": true
  }'
```

Restrict retrieval to specific documents:

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What does this document say about the methodology?",
    "document_ids": ["your-document-id"],
    "return_sources": true,
    "return_evidence": true
  }'
```

A typical response contains an answer, sources and evidence snippets.

---

## Document management

List indexed documents:

```bash
curl http://localhost:8000/documents
```

Delete a document:

```bash
curl -X DELETE http://localhost:8000/documents/your-document-id
```

If `API_KEY` is configured:

```bash
curl -X DELETE http://localhost:8000/documents/your-document-id \
  -H "X-API-Key: your_key_here"
```

Deletion removes the registry entry, the raw file and the document’s chunks from ChromaDB.

---

## Debug retrieval

Use `/debug/retrieve` to inspect retrieval before generation:

```bash
curl -X POST http://localhost:8000/debug/retrieve \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_key_here" \
  -d '{
    "question": "What are the main findings?"
  }'
```

The debug response includes retrieval settings, candidate counts, retrieved chunk previews, distances and rerank scores when reranking is enabled.

When `API_KEY` is set, `/debug/retrieve` is protected because it can expose chunk previews and retrieval internals.

---

## Retrieval tuning

`FETCH_K` controls how many candidates are retrieved before filtering and diversification. `TOP_K` controls how many final chunks are used as context.

```dotenv
TOP_K=15
FETCH_K=50
```

MMR reduces repeated or overly similar chunks:

```dotenv
MMR_ENABLED=TRUE
MMR_LAMBDA=0.65
```

HyDE can improve semantic retrieval by generating a short hypothetical answer before retrieval, but it adds one extra LLM call:

```dotenv
HYDE_ENABLED=FALSE
```

Cross-encoder reranking can improve ordering when dense retrieval returns relevant but imperfectly ranked chunks:

```dotenv
RERANKER_ENABLED=TRUE
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
RERANK_TOP_N=15
```

Reranking is disabled by default because it loads an additional model and increases latency.

---

## Reset and re-index

Rebuild the vector index whenever embedding or chunking settings change:

```text
EMBEDDING_MODEL
EMBEDDING_QUERY_PREFIX
EMBEDDING_DOCUMENT_PREFIX
CHUNK_SIZE_CHARS
CHUNK_OVERLAP_CHARS
CHUNK_SOURCE_PREFIX_ENABLED
```

Reset:

```bash
docker compose down
rm -rf data/chroma data/metadata
docker compose up -d --build
curl -X POST http://localhost:8000/ingest
```

Changing reranker settings does not require re-indexing because reranking happens after retrieval.

---

## Evaluation and tests

RagPi includes unit/API tests and a lightweight local RAG evaluation suite.

Run pytest:

```bash
python -m pytest -q
```

Run the RAG evaluation:

```bash
python eval/run_eval.py --save-results eval/results/latest.json
```

The evaluation calls the running `/query` endpoint and checks whether answers are grounded, sources and evidence are returned, unsupported questions are refused, numeric facts can be extracted and responses are not empty.

The evaluation is local-first and does not use external LLM judges or paid APIs.

---

## Troubleshooting

If Docker cannot reach Ollama, make sure `docker-compose.yml` includes:

```yaml
extra_hosts:
  - "host.docker.internal:host-gateway"
```

and `.env` contains:

```dotenv
OLLAMA_BASE_URL=http://host.docker.internal:11434
```

If `/debug/retrieve` works but `/query` fails, retrieval is probably working and the issue is generation. Check Ollama directly:

```bash
curl http://localhost:11434/api/tags
```

If answers are irrelevant, inspect `/debug/retrieve` and tune `TOP_K`, `FETCH_K`, MMR, HyDE or reranking. Avoid setting a strict `DISTANCE_THRESHOLD` until you have inspected actual distances.

If you remove documents manually from `data/raw`, reset `data/chroma` and `data/metadata` before re-ingesting so stale vectors do not remain.

---

## Current limitations

RagPi works best for semantic questions, document-specific facts, methodology questions, summaries, evidence-backed answers and filtered queries over selected documents.

Exact-value extraction can still be harder for phone numbers, emails, URLs, identifiers and similar values. These cases may benefit from a future hybrid retrieval layer combining dense retrieval with lexical or pattern-based search.

This project is intended as a local document-grounded RAG assistant and learning project. It is not a production authentication system, hosted document platform or formal benchmark.

---

## Security notes

For local development, `API_KEY` can be empty.

Before exposing the API beyond localhost, configure `API_KEY`, restrict CORS origins, avoid exposing personal documents publicly, and consider adding rate limiting or an authentication proxy.

Admin and write operations should remain protected. Debug retrieval should also be protected because it can expose retrieved document previews.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Credits

This project was created as a practical local RAG assistant for working with personal and academic documents. It was especially useful for exploring large, dense and technical files by making it easier to ask grounded questions, retrieve relevant evidence and study complex material more efficiently.

Beyond being a software project, RagPi served as a hands-on learning experience in document ingestion, vector search, local LLMs, retrieval quality, evidence grounding, API design and evaluation.