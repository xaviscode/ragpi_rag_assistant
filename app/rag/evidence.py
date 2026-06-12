from __future__ import annotations

import re
from typing import Any, Callable


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    cleaned = []
    for p in parts:
        p = re.sub(r"\s+", " ", p).strip()
        if _is_useful_short_evidence(p):
            cleaned.append(p)
    return cleaned


def _is_useful_short_evidence(text: str) -> bool:
    text = text.strip()
    if not text:
        return False
    if re.search(r"\d", text):
        return True
    if re.search(r"[@€$£%]", text):
        return True
    if 10 <= len(text) < 40 and len(text.split()) >= 2:
        return True
    return len(text) >= 40


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def select_evidence(
    question: str,
    retrieved_items: list[dict[str, Any]],
    embed_query_fn: Callable[[str], list[float]],
    embed_texts_fn: Callable[[list[str]], list[list[float]]],
    max_evidence: int = 10,
    max_total_chars: int = 3800,
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []

    for item in retrieved_items:
        text = item.get("text") or ""
        sentences = _split_sentences(text)
        if not sentences and text.strip():
            sentences = [text.strip()[:600]]
        for sentence in sentences:
            candidates.append(
                {
                    "source": item.get("source", "unknown"),
                    "chunk_id": item.get("chunk_id", ""),
                    "quote": sentence,
                }
            )

    if not candidates:
        return {"evidence": [], "evidence_text": ""}

    candidates = candidates[:80]
    qemb = embed_query_fn(question)
    embs = embed_texts_fn([c["quote"] for c in candidates])
    scored = sorted(
        [(_dot(qemb, emb), cand) for cand, emb in zip(candidates, embs)],
        key=lambda x: x[0],
        reverse=True,
    )

    evidence = []
    seen = set()
    total_chars = 0

    for _, cand in scored:
        key = (cand["source"], cand["quote"][:120])
        if key in seen:
            continue
        if total_chars + len(cand["quote"]) > max_total_chars:
            break
        evidence.append(cand)
        seen.add(key)
        total_chars += len(cand["quote"])
        if len(evidence) >= max_evidence:
            break

    lines = [
        f"[{i}] Source: {ev['source']} | Chunk: {ev['chunk_id']}\n{ev['quote']}"
        for i, ev in enumerate(evidence, start=1)
    ]
    return {"evidence": evidence, "evidence_text": "\n\n".join(lines)}
