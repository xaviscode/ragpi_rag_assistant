def build_prompt_parts(evidence_text: str, question: str):
    question_prompt = f"""You are a helpful assistant.

RULES:
- You must answer using ONLY the EVIDENCE lines provided.
- If the answer is not in the context, say: "I don't know based on the provided documents."
- Match the user's intent:
  * If the user asks for a name, ID, date, or single fact: answer in ONE line.
  * If the user asks to summarize/explain/overview: write a detailed summary (6-12 sentences), grouped in paragraphs or bullets.
  * If the user asks for steps: return a numbered list.
- Do not be overly brief unless the question clearly requires it.

Question: {question}

EVIDENCE:
"""
    return question_prompt, evidence_text