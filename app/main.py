"""
FastAPI backend.

Endpoints:
  POST /ingest         - chunk + embed + load all PDFs in data/sample_docs
  POST /query          - retrieve + prompt + generate an answer
  GET  /health         - basic health check
"""
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi import UploadFile, File
from typing import List
from app.chunking import chunk_directory
from app.config import settings
from app.prompts import Strategy, build_prompt
from app.providers import Provider, call_llm
from app.vectorstore import get_client, search, upsert_chunks

app = FastAPI(title="MedDocQA", version="0.1.0")

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "sample_docs"


class QueryRequest(BaseModel):
    question: str
    strategy: Strategy = Strategy.ZERO_SHOT
    provider: Provider = Provider.GEMINI
    top_k: int | None = None


class QueryResponse(BaseModel):
    answer: str
    strategy: str
    provider: str
    model: str
    latency_seconds: float
    estimated_cost_usd: float
    sources: list[dict]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ingest")
def ingest():
    chunks = chunk_directory(DATA_DIR)
    if not chunks:
        raise HTTPException(
            status_code=400,
            detail=f"No PDFs found in {DATA_DIR}. Add some and retry.",
        )
    client = get_client()
    count = upsert_chunks(client, chunks)
    return {"chunks_loaded": count, "source_dir": str(DATA_DIR)}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    client = get_client()
    retrieved = search(client, req.question, top_k=req.top_k)

    if not retrieved:
        raise HTTPException(
            status_code=400,
            detail="No documents indexed yet. Call /ingest first.",
        )

    prompt = build_prompt(req.strategy, req.question, retrieved)
    result = call_llm(prompt, req.provider)

    return QueryResponse(
        answer=result.text,
        strategy=req.strategy.value,
        provider=result.provider,
        model=result.model,
        latency_seconds=result.latency_seconds,
        estimated_cost_usd=result.estimated_cost_usd,
        sources=[
            {"source": c["source"], "page": c["page"], "score": round(c["score"], 3)}
            for c in retrieved
        ],
    )
@app.post("/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    saved = []
    for file in files:
        if not file.filename.endswith(".pdf"):
            continue
        dest = DATA_DIR / file.filename
        content = await file.read()
        dest.write_bytes(content)
        saved.append(file.filename)
    return {"uploaded": saved, "count": len(saved)}