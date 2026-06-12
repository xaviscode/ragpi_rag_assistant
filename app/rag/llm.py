from __future__ import annotations

import requests


class LocalLLM:
    def __init__(self, model_name: str, base_url: str, timeout: int = 120, keep_alive: str = "10m"):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.keep_alive = keep_alive

    def generate(self, system_prompt: str, user_prompt: str, max_new_tokens: int, temperature: float) -> str:
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model_name,
            "system": system_prompt.strip(),
            "prompt": user_prompt.strip(),
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": temperature,
                "num_predict": max_new_tokens,
            },
        }
        r = requests.post(url, json=payload, timeout=self.timeout)
        if not r.ok:
            print("OLLAMA ERROR STATUS:", r.status_code)
            print("OLLAMA ERROR BODY:", r.text[:4000])
            r.raise_for_status()
        data = r.json()
        response = (data.get("response") or "").strip()
        if not response:
            print("OLLAMA EMPTY RESPONSE:", data)
        return response

    def generate_simple(self, prompt: str, max_new_tokens: int = 120, temperature: float = 0.1) -> str:
        return self.generate(
            system_prompt="You are a concise assistant.",
            user_prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
