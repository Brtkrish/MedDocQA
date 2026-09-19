"""
Qdrant wrapper - thin layer over the client so the rest of the app doesn't
need to know Qdrant's API shape.
"""
from typing import List

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from app.chunking import Chunk
from app.config import settings
from app.embeddings import embed_texts, embed_query


def get_client() -> QdrantClient:
    return QdrantClient(
        url=settings.QDRANT_URL,
        api_key=settings.QDRANT_API_KEY or None,
    )


def ensure_collection(client: QdrantClient) -> None:
    collections = [c.name for c in client.get_collections().collections]
    if settings.COLLECTION_NAME not in collections:
        client.create_collection(
            collection_name=settings.COLLECTION_NAME,
            vectors_config=qmodels.VectorParams(
                size=settings.EMBEDDING_DIM,
                distance=qmodels.Distance.COSINE,
            ),
        )


def upsert_chunks(client: QdrantClient, chunks: List[Chunk]) -> int:
    """Embed and load a list of chunks into Qdrant. Returns count loaded."""
    if not chunks:
        return 0

    ensure_collection(client)
    texts = [c.text for c in chunks]
    vectors = embed_texts(texts)

    points = [
        qmodels.PointStruct(
            id=i,
            vector=vectors[i].tolist(),
            payload={
                "chunk_id": chunk.chunk_id,
                "source": chunk.source,
                "page": chunk.page,
                "text": chunk.text,
            },
        )
        for i, chunk in enumerate(chunks)
    ]
    client.upsert(collection_name=settings.COLLECTION_NAME, points=points)
    return len(points)


def search(client: QdrantClient, query: str, top_k: int = None) -> List[dict]:
    """Return top_k most relevant chunks (as payload dicts) for a query."""
    top_k = top_k or settings.TOP_K
    query_vector = embed_query(query).tolist()
    # qdrant-client >=1.10 renamed the old .search() to .query_points();
    # .search() still exists in some versions but is deprecated/removed in
    # others, so we use the current API directly.
    results = client.query_points(
        collection_name=settings.COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
    ).points
    return [
        {**hit.payload, "score": hit.score}
        for hit in results
    ]
