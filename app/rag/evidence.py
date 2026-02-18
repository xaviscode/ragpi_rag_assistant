from __future__ import annotations
import re
from typing import Any, Dict, List

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\[])")

def _split_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", (text or "")).strip()
    if not text:
        return []
    if sum(text.count(p) for p in ".!?") == 0:
        return [text]
    sents = _SENT_SPLIT.split(text)
    out = []
    for s in sents:
        s = s.strip()
        if len(s) < 20:
            continue
        out.append(s)
    return out or [text]

def select_evidence(
    question: str,
    retrieved_items: List[Dict[str, Any]],
    embed_query_fn,
    embed_texts_fn,
    max_evidence: int = 10,
    max_total_chars: int = 3200,
) -> Dict[str, Any]:
    candidates: List[Dict[str, Any]] = []
    for it in retrieved_items:
        for sent in _split_sentences(it.get("text", "")):
            candidates.append({
                "source": it.get("source", "unknown"),
                "chunk_id": it.get("chunk_id", ""),
                "quote": sent,
            })

    if not candidates:
        return {"evidence": [], "evidence_text": ""}

    qemb = embed_query_fn(question)
    texts = [c["quote"] for c in candidates]
    embs = embed_texts_fn(texts)

    scored = []
    for c, e in zip(candidates, embs):
        score = sum(a*b for a, b in zip(qemb, e))
        scored.append((score, c))
    scored.sort(key=lambda x: x[0], reverse=True)

    seen = set()
    evidence = []
    total = 0
    for score, c in scored:
        q = c["quote"].strip()
        if q in seen:
            continue
        seen.add(q)
        if len(q) > 520:
            q = q[:520].rstrip() + "…"
        if total + len(q) + 40 > max_total_chars:
            continue
        evidence.append({"source": c["source"], "chunk_id": c["chunk_id"], "quote": q})
        total += len(q) + 1
        if len(evidence) >= max_evidence:
            break

    lines = [f"[{i}] ({ev['source']}) \"{ev['quote']}\"" for i, ev in enumerate(evidence, 1)]
    return {"evidence": evidence, "evidence_text": "\n".join(lines)}
