"""
Document parsing and chunking - built from scratch so you can explain
exactly how retrieval context gets built, rather than pointing at a
framework and saying "it handles it."
"""
from dataclasses import dataclass
from pathlib import Path
from typing import List

from pypdf import PdfReader

from app.config import settings


@dataclass
class Chunk:
    chunk_id: str
    source: str       # filename
    page: int         # 1-indexed page number this chunk starts on
    text: str


def extract_text_by_page(pdf_path: Path) -> List[str]:
    """Return a list of raw text strings, one per page."""
    reader = PdfReader(str(pdf_path))
    return [page.extract_text() or "" for page in reader.pages]


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    Simple sliding-window character chunker.
    Not the fanciest strategy (no sentence-boundary awareness), but
    transparent, debuggable, and good enough for a first RAG pass -
    which is itself a defensible design decision to discuss.
    """
    if not text.strip():
        return []

    chunks = []
    start = 0
    step = max(chunk_size - overlap, 1)
    while start < len(text):
        chunk = text[start : start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        start += step
    return chunks


def chunk_pdf(pdf_path: Path) -> List[Chunk]:
    """Parse a PDF and return a flat list of Chunk objects with metadata."""
    pages = extract_text_by_page(pdf_path)
    chunks: List[Chunk] = []
    for page_num, page_text in enumerate(pages, start=1):
        for i, piece in enumerate(
            chunk_text(page_text, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)
        ):
            chunks.append(
                Chunk(
                    chunk_id=f"{pdf_path.stem}_p{page_num}_c{i}",
                    source=pdf_path.name,
                    page=page_num,
                    text=piece,
                )
            )
    return chunks


def chunk_directory(directory: Path) -> List[Chunk]:
    """Chunk every PDF in a directory."""
    all_chunks: List[Chunk] = []
    for pdf_path in sorted(directory.glob("*.pdf")):
        all_chunks.extend(chunk_pdf(pdf_path))
    return all_chunks
