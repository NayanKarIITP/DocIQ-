# """retrieval/vector_store.py — fixed for qdrant-client >= 1.7 (query_points API)"""
# from loguru import logger
# from qdrant_client import QdrantClient
# from qdrant_client.models import (
#     Distance, FieldCondition, Filter,
#     MatchValue, PointStruct, VectorParams,
# )
# from config import settings


# class VectorStore:
#     def __init__(self):
#         self.client     = QdrantClient(":memory:")
#         self.collection = settings.qdrant_collection
#         self.dim        = settings.embedding_dim
#         self._created   = False

#     def create_collection(self, recreate: bool = True) -> None:
#         if self._created and not recreate:
#             logger.info(f"Collection '{self.collection}' already exists (in-memory)")
#             return
#         try:
#             self.client.delete_collection(self.collection)
#         except Exception:
#             pass
#         self.client.create_collection(
#             collection_name=self.collection,
#             vectors_config=VectorParams(size=self.dim, distance=Distance.COSINE),
#         )
#         self._created = True
#         logger.success(f"In-memory collection '{self.collection}' ready (dim={self.dim})")

#     def upsert(self, chunks_with_embeddings: list) -> int:
#         if not chunks_with_embeddings:
#             return 0
#         if not self._created:
#             self.create_collection()
#         points = []
#         for chunk, embedding in chunks_with_embeddings:
#             point_id = int(chunk.chunk_id[:8], 16)
#             points.append(PointStruct(
#                 id=point_id,
#                 vector=list(embedding),
#                 payload={
#                     "chunk_id":    chunk.chunk_id,
#                     "doc_id":      chunk.doc_id,
#                     "filename":    chunk.filename,
#                     "page_number": chunk.page_number,
#                     "chunk_type":  chunk.chunk_type.value,
#                     "content":     chunk.content,
#                     "has_image":   chunk.image_b64 is not None,
#                     "metadata":    chunk.metadata,
#                 },
#             ))
#         for i in range(0, len(points), 100):
#             self.client.upsert(
#                 collection_name=self.collection,
#                 points=points[i:i+100],
#                 wait=True,
#             )
#         logger.success(f"Upserted {len(points)} chunks (in-memory)")
#         return len(points)

#     def search(self, query_embedding: list, top_k: int = None,
#                filter_doc_id: str = None, filter_chunk_type: str = None) -> list:
#         if not self._created:
#             return []
#         top_k = top_k or settings.top_k_retrieval

#         must = []
#         if filter_doc_id:
#             must.append(FieldCondition(key="doc_id",      match=MatchValue(value=filter_doc_id)))
#         if filter_chunk_type:
#             must.append(FieldCondition(key="chunk_type",  match=MatchValue(value=filter_chunk_type)))
#         query_filter = Filter(must=must) if must else None

#         # Support both old and new Qdrant client API
#         try:
#             # New API (>= 1.7): query_points
#             results = self.client.query_points(
#                 collection_name=self.collection,
#                 query=list(query_embedding),
#                 limit=top_k,
#                 query_filter=query_filter,
#                 with_payload=True,
#             ).points
#             return [{**h.payload, "score": h.score} for h in results]
#         except AttributeError:
#             # Old API: search
#             results = self.client.search(
#                 collection_name=self.collection,
#                 query_vector=list(query_embedding),
#                 limit=top_k,
#                 query_filter=query_filter,
#                 with_payload=True,
#             )
#             return [{**h.payload, "score": h.score} for h in results]

#     def delete_document(self, doc_id: str) -> None:
#         if not self._created:
#             return
#         self.client.delete(
#             collection_name=self.collection,
#             points_selector=Filter(
#                 must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
#             ),
#         )

#     def count(self) -> int:
#         if not self._created:
#             return 0
#         return self.client.count(collection_name=self.collection).count

#     def list_documents(self) -> list:
#         if not self._created:
#             return []
#         seen   = {}
#         offset = None
#         while True:
#             results, offset = self.client.scroll(
#                 collection_name=self.collection,
#                 limit=100, offset=offset,
#                 with_payload=["doc_id", "filename"],
#             )
#             for p in results:
#                 did = p.payload["doc_id"]
#                 if did not in seen:
#                     seen[did] = p.payload["filename"]
#             if offset is None:
#                 break
#         return [{"doc_id": k, "filename": v} for k, v in seen.items()]




"""
retrieval/vector_store.py

Persistent Qdrant vector store for DocIQ.

Qdrant runs locally at:
    http://localhost:6333

Unlike the previous implementation, this file does NOT use
QdrantClient(":memory:"). Therefore vectors persist even when
FastAPI is restarted.
"""

from loguru import logger

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from config import settings


class VectorStore:

    def __init__(self):

        # ==========================================================
        # QDRANT CONNECTION
        # ==========================================================
        if getattr(settings, "qdrant_url", None):
            self.client = QdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key or None,
            )
            logger.info(f"Qdrant client initialized (Cloud/URL): {settings.qdrant_url}")
        else:
            self.host = getattr(settings, "qdrant_host", "localhost")
            self.port = getattr(settings, "qdrant_port", 6333)
            self.client = QdrantClient(
                host=self.host,
                port=self.port,
                api_key=getattr(settings, "qdrant_api_key", None) or None,
            )
            logger.info(f"Qdrant client initialized: http://{self.host}:{self.port}")

        self.collection = settings.qdrant_collection
        self.dim = settings.embedding_dim
        self._created = False

    # ==============================================================
    # CREATE / CHECK COLLECTION
    # ==============================================================

    def create_collection(self, recreate: bool = False) -> None:
        """
        Create the collection if it does not exist.

        IMPORTANT:
        recreate=False by default.

        This prevents the collection from being deleted every time
        FastAPI starts.
        """

        try:

            # ------------------------------------------------------
            # Check whether collection already exists
            # ------------------------------------------------------

            collections = self.client.get_collections()

            existing_collections = {
                collection.name
                for collection in collections.collections
            }

            # ------------------------------------------------------
            # Collection already exists
            # ------------------------------------------------------

            if self.collection in existing_collections:

                if recreate:

                    logger.warning(
                        f"Recreating Qdrant collection "
                        f"'{self.collection}'..."
                    )

                    self.client.delete_collection(
                        collection_name=self.collection
                    )

                else:

                    self._created = True

                    logger.success(
                        f"Qdrant collection "
                        f"'{self.collection}' already exists "
                        f"(dim={self.dim})"
                    )

                    return

            # ------------------------------------------------------
            # Create collection
            # ------------------------------------------------------

            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(
                    size=self.dim,
                    distance=Distance.COSINE,
                ),
            )

            self._created = True

            logger.success(
                f"Qdrant collection "
                f"'{self.collection}' ready "
                f"(dim={self.dim})"
            )

        except Exception as e:

            logger.error(
                f"Failed to create/check Qdrant collection: {e}"
            )

            raise

    # ==============================================================
    # UPSERT
    # ==============================================================

    def upsert(
        self,
        chunks_with_embeddings: list
    ) -> int:
        """
        Insert document chunks and their embeddings into Qdrant.
        """

        if not chunks_with_embeddings:

            logger.warning(
                "No chunks with embeddings to upsert."
            )

            return 0

        # Make sure collection exists.
        if not self._created:

            self.create_collection(
                recreate=False
            )

        points = []

        for chunk, embedding in chunks_with_embeddings:

            try:

                # --------------------------------------------------
                # Convert chunk ID into deterministic integer ID
                # --------------------------------------------------

                point_id = int(
                    chunk.chunk_id[:8],
                    16
                )

                # --------------------------------------------------
                # Create Qdrant point
                # --------------------------------------------------

                point = PointStruct(

                    id=point_id,

                    vector=list(embedding),

                    payload={

                        "chunk_id": chunk.chunk_id,

                        "doc_id": chunk.doc_id,

                        "filename": chunk.filename,

                        "page_number": chunk.page_number,

                        "chunk_type": (
                            chunk.chunk_type.value
                        ),

                        "content": chunk.content,

                        "has_image": (
                            chunk.image_b64 is not None
                        ),

                        "metadata": chunk.metadata,
                    },
                )

                points.append(point)

            except Exception as e:

                logger.error(
                    f"Failed to prepare chunk "
                    f"{getattr(chunk, 'chunk_id', 'unknown')}: {e}"
                )

        if not points:

            logger.warning(
                "No valid points were prepared for Qdrant."
            )

            return 0

        # ----------------------------------------------------------
        # Batch upsert
        # ----------------------------------------------------------

        for i in range(
            0,
            len(points),
            100
        ):

            batch = points[
                i:i + 100
            ]

            self.client.upsert(

                collection_name=self.collection,

                points=batch,

                wait=True,
            )

        logger.success(
            f"Upserted {len(points)} chunks "
            f"into Qdrant collection "
            f"'{self.collection}'"
        )

        return len(points)

    # ==============================================================
    # SEARCH
    # ==============================================================

    def search(
        self,
        query_embedding: list,
        top_k: int = None,
        filter_doc_id: str = None,
        filter_chunk_type: str = None,
    ) -> list:
        """
        Search Qdrant using cosine similarity.
        """

        if not query_embedding:

            logger.warning(
                "Empty query embedding."
            )

            return []

        # ----------------------------------------------------------
        # Make sure collection exists
        # ----------------------------------------------------------

        if not self._created:

            try:

                self.create_collection(
                    recreate=False
                )

            except Exception as e:

                logger.error(
                    f"Unable to initialize Qdrant "
                    f"collection before search: {e}"
                )

                return []

        # ----------------------------------------------------------
        # Number of results
        # ----------------------------------------------------------

        top_k = (
            top_k
            or settings.top_k_retrieval
        )

        # ----------------------------------------------------------
        # Build filters
        # ----------------------------------------------------------

        must_conditions = []

        if filter_doc_id:

            must_conditions.append(
                FieldCondition(
                    key="doc_id",
                    match=MatchValue(
                        value=filter_doc_id
                    ),
                )
            )

        if filter_chunk_type:

            must_conditions.append(
                FieldCondition(
                    key="chunk_type",
                    match=MatchValue(
                        value=filter_chunk_type
                    ),
                )
            )

        query_filter = (
            Filter(
                must=must_conditions
            )
            if must_conditions
            else None
        )

        # ==========================================================
        # QDRANT QUERY
        # ==========================================================

        try:

            # ------------------------------------------------------
            # New qdrant-client API
            # ------------------------------------------------------

            response = self.client.query_points(

                collection_name=self.collection,

                query=list(query_embedding),

                limit=top_k,

                query_filter=query_filter,

                with_payload=True,
            )

            results = response.points

            formatted_results = []

            for point in results:

                payload = point.payload or {}

                formatted_results.append(
                    {
                        **payload,
                        "score": point.score,
                    }
                )

            logger.info(
                f"Qdrant search returned "
                f"{len(formatted_results)} results"
            )

            return formatted_results

        except AttributeError:

            # ------------------------------------------------------
            # Fallback for older qdrant-client versions
            # ------------------------------------------------------

            try:

                results = self.client.search(

                    collection_name=self.collection,

                    query_vector=list(
                        query_embedding
                    ),

                    limit=top_k,

                    query_filter=query_filter,

                    with_payload=True,
                )

                formatted_results = []

                for point in results:

                    payload = point.payload or {}

                    formatted_results.append(
                        {
                            **payload,
                            "score": point.score,
                        }
                    )

                return formatted_results

            except Exception as e:

                logger.error(
                    f"Qdrant search failed: {e}"
                )

                return []

        except Exception as e:

            logger.error(
                f"Qdrant query failed: {e}"
            )

            return []

    # ==============================================================
    # DELETE DOCUMENT
    # ==============================================================

    def delete_document(
        self,
        doc_id: str
    ) -> None:
        """
        Delete all chunks belonging to a document.
        """

        if not self._created:

            self.create_collection(
                recreate=False
            )

        try:

            self.client.delete(

                collection_name=self.collection,

                points_selector=Filter(

                    must=[
                        FieldCondition(

                            key="doc_id",

                            match=MatchValue(
                                value=doc_id
                            ),
                        )
                    ]
                ),

                wait=True,
            )

            logger.success(
                f"Deleted document "
                f"{doc_id} from Qdrant"
            )

        except Exception as e:

            logger.error(
                f"Failed to delete document "
                f"{doc_id}: {e}"
            )

    def delete_all(self) -> None:
        """Clear all vectors from the collection."""
        try:
            self.create_collection(recreate=True)
            logger.success(f"Cleared all vectors from '{self.collection}'")
        except Exception as e:
            logger.error(f"Failed to clear all vectors: {e}")

    # ==============================================================
    # COUNT
    # ==============================================================

    def count(self) -> int:
        """
        Return number of vectors currently stored.
        """

        if not self._created:

            try:

                self.create_collection(
                    recreate=False
                )

            except Exception:

                return 0

        try:

            result = self.client.count(
                collection_name=self.collection
            )

            return result.count

        except Exception as e:

            logger.error(
                f"Failed to count Qdrant points: {e}"
            )

            return 0

    # ==============================================================
    # LIST DOCUMENTS
    # ==============================================================

    def list_documents(self) -> list:
        """
        Return unique documents stored in Qdrant.
        """

        if not self._created:

            try:

                self.create_collection(
                    recreate=False
                )

            except Exception:

                return []

        seen = {}

        offset = None

        try:

            while True:

                results, offset = self.client.scroll(

                    collection_name=self.collection,

                    limit=100,

                    offset=offset,

                    with_payload=[
                        "doc_id",
                        "filename",
                    ],
                )

                for point in results:

                    payload = point.payload or {}

                    doc_id = payload.get(
                        "doc_id"
                    )

                    filename = payload.get(
                        "filename"
                    )

                    if (
                        doc_id
                        and doc_id not in seen
                    ):

                        seen[doc_id] = filename

                if offset is None:

                    break

            return [
                {
                    "doc_id": doc_id,
                    "filename": filename,
                }

                for doc_id, filename
                in seen.items()
            ]

        except Exception as e:

            logger.error(
                f"Failed to list Qdrant documents: {e}"
            )

            return []

    # ==============================================================
    # GET ALL CHUNKS (FOR BM25 HYBRID SEARCH)
    # ==============================================================

    def get_all_chunks(self, limit: int = 10000) -> list[dict]:
        """
        Scroll all chunk payloads from Qdrant for keyword / BM25 indexing.
        """
        if not self._created:
            try:
                self.create_collection(recreate=False)
            except Exception:
                return []

        all_chunks = []
        offset = None

        try:
            while True:
                results, offset = self.client.scroll(
                    collection_name=self.collection,
                    limit=min(limit, 500),
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )
                for point in results:
                    payload = point.payload or {}
                    all_chunks.append(dict(payload))

                if offset is None or len(all_chunks) >= limit:
                    break

            return all_chunks
        except Exception as e:
            logger.error(f"Failed to scroll all Qdrant chunks: {e}")
            return []

    # ==============================================================
    # HEALTH CHECK
    # ==============================================================

    def health_check(self) -> bool:
        """
        Check whether Qdrant is reachable.
        """

        try:

            self.client.get_collections()

            logger.info(
                "Qdrant health check: OK"
            )

            return True

        except Exception as e:

            logger.error(
                f"Qdrant health check failed: {e}"
            )

            return False