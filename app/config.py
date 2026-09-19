"""
Central configuration for MedDocQA.
Loads all settings from environment variables (.env) so nothing is hardcoded.
"""
import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # Vector DB
    QDRANT_URL: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    QDRANT_API_KEY: str = os.getenv("QDRANT_API_KEY", "")
    COLLECTION_NAME: str = "meddocqa_chunks"

    # Embedding model - runs locally on PyTorch via sentence-transformers
    EMBEDDING_MODEL: str = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    EMBEDDING_DIM: int = 384  # all-MiniLM-L6-v2 output size

    # Chunking
    CHUNK_SIZE: int = 500       # characters per chunk
    CHUNK_OVERLAP: int = 80     # overlap between consecutive chunks
    TOP_K: int = 4              # chunks retrieved per query

    # LLM providers
    # LLM providers
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL_NAME", "gemini-3.6-flash")

    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL_NAME", "openai/gpt-oss-20b")

    # Rough per-1K-token USD pricing, used only to *estimate* cost in the
    # benchmark/eval output. Update these if pricing changes.
    PRICING = {
        "gemini-3.6-flash": {"input": 0.000075, "output": 0.0003},
        "openai/gpt-oss-20b": {"input": 0.00005, "output": 0.00008},
    }

settings = Settings()
