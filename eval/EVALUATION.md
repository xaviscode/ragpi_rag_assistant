# RAG Evaluation Plan

This project includes a lightweight local evaluation suite for checking whether RagPi answers questions from uploaded documents in a grounded, reproducible and testable way.

The evaluation is intentionally simple. It does not use external LLM judges or paid APIs. Instead, it sends predefined questions to the running `/query` endpoint and checks the returned answer, sources and evidence against expected patterns.

## Scope

The evaluation focuses on document-grounded RAG behaviour. It checks whether the assistant can retrieve relevant chunks, answer from evidence, refuse unsupported questions, extract factual details, handle numeric values and avoid empty responses.

The current thesis-focused evaluation uses `FCI_Thesis.pdf` as the controlled test document. The questions cover document identity, objectives, indicator design, methodology, validation, empirical results, limitations and grounded refusal.

## Files

```text
eval/
  EVALUATION.md
  questions.json
  run_eval.py
  results/
```

`questions.json` contains the evaluation cases. Each case defines a question, expected answer patterns, expected evidence patterns and whether sources or evidence are required.

`run_eval.py` sends each case to the API, evaluates the response and prints a summary. Results can also be saved as JSON for comparison across runs.

## Running the evaluation

Start the API first:

```bash
docker compose up -d --build
```

Then run:

```bash
python eval/run_eval.py --save-results eval/results/latest.json
```

Run the unit tests separately:

```bash
python -m pytest -q
```

## Interpreting results

A passing case means the answer satisfied the configured checks. A failing case does not always mean the system is wrong. Failures usually fall into one of four categories: retrieval missed the best chunk, evidence was relevant but did not match the expected string, the answer was correct but phrased differently, or the test was too strict.

The most important cases are grounded refusal, document identity, methodology retrieval, numeric extraction, source presence, evidence presence and non-empty answers.

## Current evaluation goals

The main target is not perfect accuracy on every question. The goal is to make retrieval quality visible and comparable after each change.

Useful comparisons include baseline retrieval versus MMR, BGE query-prefix changes, Ollama generation changes, and optional reranking.

When retrieval settings change, save a new result file:

```bash
python eval/run_eval.py --save-results eval/results/reranker_enabled.json
```

Then compare it with the previous result.

## Re-indexing before evaluation

Rebuild the index whenever embedding or chunking settings change:

```bash
docker compose down
rm -rf data/chroma data/metadata
docker compose up -d --build
curl -X POST http://localhost:8000/ingest
```

Re-index after changing:

```text
EMBEDDING_MODEL
EMBEDDING_QUERY_PREFIX
EMBEDDING_DOCUMENT_PREFIX
CHUNK_SIZE_CHARS
CHUNK_OVERLAP_CHARS
CHUNK_SOURCE_PREFIX_ENABLED
```

Reranker settings do not require re-indexing because reranking happens after retrieval.

## Notes

This evaluation suite is designed for development and portfolio validation. It is not a benchmark against external datasets, and it does not prove production-grade reliability. Its purpose is to make the RAG system easier to test, debug and improve in a controlled local setting.