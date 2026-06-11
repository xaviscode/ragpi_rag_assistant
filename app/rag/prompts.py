from __future__ import annotations


def build_prompt_parts(evidence_text: str, question: str) -> tuple[str, str]:
    system_prompt = """You are a document-grounded assistant.

Rules:
- Answer only using the provided evidence.
- If the evidence is insufficient, say: "I don't know based on the provided documents."
- Do not use outside knowledge.
- Be concise and factual.
- When useful, mention the source document names.
"""
    user_prompt = f"""Evidence:
{evidence_text}

Question:
{question}

Answer:
"""
    return system_prompt, user_prompt
