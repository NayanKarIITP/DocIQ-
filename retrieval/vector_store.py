"""retrieval/vector_store.py — fixed for qdrant-client >= 1.7 (query_points API)"""
from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, FieldCondition, Filter,
    MatchValue, PointStruct, VectorParams,
)
from config import settings


class VectorStore:
    def __init__(self):
        self.client     = QdrantClient(":memory:")
        self.collection = settings.qdrant_collection
        self.dim        = settings.embedding_dim
        self._created   = False

    def create_collection(self, recreate: bool = True) -> None:
        if self._created and not recreate:
            logger.info(f"Collection '{self.collection}' already exists (in-memory)")
            return
        try:
            self.client.delete_collection(self.collection)
        except Exception:
            pass
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(size=self.dim, distance=Distance.COSINE),
        )
        self._created = True
        logger.success(f"In-memory collection '{self.collection}' ready (dim={self.dim})")

    def upsert(self, chunks_with_embeddings: list) -> int:
        if not chunks_with_embeddings:
            return 0
        if not self._created:
            self.create_collection()
        points = []
        for chunk, embedding in chunks_with_embeddings:
            point_id = int(chunk.chunk_id[:8], 16)
            points.append(PointStruct(
                id=point_id,
                vector=list(embedding),
                payload={
                    "chunk_id":    chunk.chunk_id,
                    "doc_id":      chunk.doc_id,
                    "filename":    chunk.filename,
                    "page_number": chunk.page_number,
                    "chunk_type":  chunk.chunk_type.value,
                    "content":     chunk.content,
                    "has_image":   chunk.image_b64 is not None,
                    "metadata":    chunk.metadata,
                },
            ))
        for i in range(0, len(points), 100):
            self.client.upsert(
                collection_name=self.collection,
                points=points[i:i+100],
                wait=True,
            )
        logger.success(f"Upserted {len(points)} chunks (in-memory)")
        return len(points)

    def search(self, query_embedding: list, top_k: int = None,
               filter_doc_id: str = None, filter_chunk_type: str = None) -> list:
        if not self._created:
            self.create_collection()
            return []
        top_k = top_k or settings.top_k_retrieval

        must = []
        if filter_doc_id:
            must.append(FieldCondition(key="doc_id",      match=MatchValue(value=filter_doc_id)))
        if filter_chunk_type:
            must.append(FieldCondition(key="chunk_type",  match=MatchValue(value=filter_chunk_type)))
        query_filter = Filter(must=must) if must else None

        # Support both old and new Qdrant client API
        try:
            # New API (>= 1.7): query_points
            results = self.client.query_points(
                collection_name=self.collection,
                query=list(query_embedding),
                limit=top_k,
                query_filter=query_filter,
                with_payload=True,
            ).points
            return [{**h.payload, "score": h.score} for h in results]
        except AttributeError:
            # Old API: search
            results = self.client.search(
                collection_name=self.collection,
                query_vector=list(query_embedding),
                limit=top_k,
                query_filter=query_filter,
                with_payload=True,
            )
            return [{**h.payload, "score": h.score} for h in results]

    def delete_document(self, doc_id: str) -> None:
        if not self._created:
            return
        self.client.delete(
            collection_name=self.collection,
            points_selector=Filter(
                must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
            ),
        )

    def count(self) -> int:
        if not self._created:
            return 0
        return self.client.count(collection_name=self.collection).count

    def list_documents(self) -> list:
        if not self._created:
            self.create_collection()
            return []
        seen   = {}
        offset = None
        while True:
            results, offset = self.client.scroll(
                collection_name=self.collection,
                limit=100, offset=offset,
                with_payload=["doc_id", "filename"],
            )
            for p in results:
                did = p.payload["doc_id"]
                if did not in seen:
                    seen[did] = p.payload["filename"]
            if offset is None:
                break
        return [{"doc_id": k, "filename": v} for k, v in seen.items()]
