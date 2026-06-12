from __future__ import annotations

import os
import re
from pathlib import Path

from bs4 import BeautifulSoup
from pypdf import PdfReader

SUPPORTED_EXTENSIONS = {".txt", ".md", ".html", ".htm", ".pdf", ".tex"}


def list_files(raw_dir: str) -> list[str]:
    if not os.path.exists(raw_dir):
        return []
    paths: list[str] = []
    for root, _, files in os.walk(raw_dir):
        for filename in files:
            path = os.path.join(root, filename)
            if Path(path).suffix.lower() in SUPPORTED_EXTENSIONS:
                paths.append(path)
    return sorted(paths)


def read_text(path: str) -> str:
    ext = Path(path).suffix.lower()
    if ext in {".txt", ".md", ".tex"}:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    if ext in {".html", ".htm"}:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            soup = BeautifulSoup(f.read(), "html.parser")
        return soup.get_text("\n")
    if ext == ".pdf":
        reader = PdfReader(path)
        pages = []
        for page in reader.pages:
            text = page.extract_text() or ""
            pages.append(text)
        return "\n".join(pages)

    raise ValueError(f"Unsupported file type: {ext}")


def clean_text(text: str) -> str:
    text = text.replace("\\x00", " ")
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _split_recursive(text: str, separators: list[str]) -> list[str]:
    if not separators:
        return [text]
    sep = separators[0]
    if sep == "":
        return list(text)
    if sep in {". ", "! ", "? "}:
        parts = re.split(rf"(?<={re.escape(sep[0])})\s+", text)
    else:
        parts = text.split(sep)
        if len(parts) > 1:
            parts = [p + sep for p in parts[:-1]] + [parts[-1]]
    if len(parts) == 1:
        return _split_recursive(text, separators[1:])
    return parts


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    text = text.strip()
    if not text:
        return []

    pieces = _split_recursive(text, ["\n\n", "\n", ". ", "! ", "? ", " ", ""])
    pieces = [p.strip() for p in pieces if p and p.strip()]
    chunks: list[str] = []
    current = ""

    for piece in pieces:
        if len(piece) > chunk_size:
            if current.strip():
                chunks.append(current.strip())
                current = ""
            start = 0
            while start < len(piece):
                end = start + chunk_size
                part = piece[start:end].strip()
                if part:
                    chunks.append(part)
                if end >= len(piece):
                    break
                start = end - overlap
            continue

        candidate = f"{current} {piece}".strip() if current else piece
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current.strip():
                chunks.append(current.strip())
            tail = chunks[-1][-overlap:].strip() if overlap > 0 and chunks else ""
            current = f"{tail} {piece}".strip() if tail else piece

    if current.strip():
        chunks.append(current.strip())

    cleaned, seen = [], set()
    for chunk in chunks:
        chunk = re.sub(r"\s+", " ", chunk).strip()
        key = chunk[:200]
        if chunk and key not in seen:
            cleaned.append(chunk)
            seen.add(key)
    return cleaned
