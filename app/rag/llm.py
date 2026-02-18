from __future__ import annotations

import os
import requests

IDK = "I don't know based on the provided documents."


class LocalLLM:
    """
    Ollama backend (runs outside Docker).
    Uses /api/generate (non-streaming).
    """

    def __init__(self, model_name: str, hf_home: str):
        self.model = model_name
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434").rstrip("/")
        self.timeout = float(os.getenv("OLLAMA_TIMEOUT", "180"))

        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=5)
            r.raise_for_status()
        except Exception as e:
            raise RuntimeError(
                f"Cannot reach Ollama at {self.base_url}. "
                f"Is 'ollama serve' running on the host? Original error: {e}"
            )

    def generate(self, question_prompt: str, context: str, max_new_tokens: int = 450, temperature: float = 0.0) -> str:
        prompt = (
            "You are a careful RAG assistant.\n"
            "Use ONLY the provided context.\n"
            f"If the answer is not in the context, say: \"{IDK}\".\n\n"
            f"{question_prompt.strip()}\n\n"
            "Context:\n"
            f"{context.strip()}\n\n"
            "Answer:"
        )

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": float(temperature),
                "num_predict": int(max_new_tokens),
            },
        }

        r = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        return (data.get("response") or "").strip()