from __future__ import annotations
import os
import re
from typing import List
from bs4 import BeautifulSoup
from pypdf import PdfReader

SUPPORTED_EXTS = {".txt", ".md", ".html", ".htm", ".pdf"}

def list_files(root: str) -> List[str]:
    out = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            ext = os.path.splitext(fn)[1].lower()
            if ext in SUPPORTED_EXTS:
                out.append(os.path.join(dirpath, fn))
    return sorted(out)

def read_text(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext in {".txt", ".md"}:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    if ext in {".html", ".htm"}:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            html = f.read()
        soup = BeautifulSoup(html, "html.parser")
        return soup.get_text("\n")
    if ext == ".pdf":
        try:
            reader = PdfReader(path, strict=False)
        except Exception as e:
            raise ValueError(f"Unreadable PDF: {os.path.basename(path)} ({e})")

        pages = []
        for idx, p in enumerate(reader.pages):
            try:
                pages.append(p.extract_text() or "")
            except Exception as e:
                print(f"[ingest] Skipping file: {os.path.basename(path)} — {e}")
                continue
        return "\n".join(pages)

    raise ValueError(f"Unsupported extension: {ext}")

def clean_text(text: str) -> str:
    # Normalize whitespace
    text = text.replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()

def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    # Paragraph-based chunking with overlap (character-level)
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: List[str] = []
    buf = ""
    for p in paras:
        if len(buf) + len(p) + 2 <= chunk_size:
            buf = (buf + "\n\n" + p).strip()
        else:
            if buf:
                chunks.append(buf)
            buf = p
            while len(buf) > chunk_size:
                chunks.append(buf[:chunk_size])
                buf = buf[chunk_size - overlap:] if overlap > 0 else buf[chunk_size:]
    if buf:
        chunks.append(buf)

    if overlap <= 0 or len(chunks) <= 1:
        return chunks

    overlapped = []
    prev_tail = ""
    for c in chunks:
        if prev_tail:
            overlapped.append((prev_tail + "\n" + c)[:chunk_size])
        else:
            overlapped.append(c[:chunk_size])
        prev_tail = c[-overlap:]
    return overlapped
