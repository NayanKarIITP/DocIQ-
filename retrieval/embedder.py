


# import time
# import base64
# import io
# from typing import Optional
# import google.generativeai as genai
# from loguru import logger
# from tenacity import retry, stop_after_attempt, wait_exponential
# from config import settings
# from ingestion.parser import ChunkType
# import PIL.Image

# class GeminiEmbedder:
#     def __init__(self):
#         genai.configure(api_key=settings.gemini_api_key, transport="rest")
#         self.embed_model = settings.gemini_embed_model
#         # Ensure the model name is formatted correctly for the GenerativeModel class
#         self.gen_model_name = settings.gemini_model if settings.gemini_model.startswith("models/") else f"models/{settings.gemini_model}"
#         self.gen_model = genai.GenerativeModel(self.gen_model_name)
#         logger.info(f"Embedder ready — embed: {self.embed_model}  gen: {self.gen_model_name}")

#     def embed_chunks(self, chunks: list) -> list:
#         results = []
#         for i, chunk in enumerate(chunks):
#             if i % 10 == 0:
#                 logger.info(f"Embedding {i}/{len(chunks)}...")
            
#             try:
#                 text = self._get_text(chunk)
#                 # If text is empty or None, provide a fallback to avoid axis 0 error
#                 if not text:
#                     text = f"Empty chunk from {chunk.filename} p.{chunk.page_number}"
                
#                 emb = self._embed(text, "retrieval_document")
                
#                 if emb:
#                     if chunk.chunk_type == ChunkType.IMAGE and not chunk.content:
#                         chunk.content = text
#                     results.append((chunk, emb))
#             except Exception as e:
#                 logger.error(f"Skipping chunk {i} due to error: {e}")
#                 continue # This prevents the "Index out of bounds" from killing the app

#             # time.sleep(0.15)
#             time.sleep(1.0)  # Gemini models have a rate limit of ~300 reqs/minute, so we need to slow down the embedding loop
            
#         logger.success(f"Embedded {len(results)}/{len(chunks)} chunks")
#         return results

#     def embed_query(self, query: str) -> list:
#         return self._embed(query, "retrieval_query")

#     def _get_text(self, chunk) -> str:
#         if chunk.chunk_type == ChunkType.TEXT:
#             return chunk.content or ""
#         elif chunk.chunk_type == ChunkType.TABLE:
#             return f"Table from {chunk.filename} p.{chunk.page_number}:\n{chunk.content}"
#         elif chunk.chunk_type == ChunkType.IMAGE:
#             return self._describe_image(chunk)
#         return str(chunk.content) if chunk.content else ""

#     def _describe_image(self, chunk) -> str:
#         # Check if content already exists to avoid unnecessary API calls
#         if chunk.content and len(chunk.content.strip()) > 10:
#             return chunk.content

#         if not chunk.image_b64:
#             return f"Image from {chunk.filename} page {chunk.page_number}"
        
#         try:
#             image_bytes = base64.b64decode(chunk.image_b64)
#             img = PIL.Image.open(io.BytesIO(image_bytes))
            
#             response = self.gen_model.generate_content([
#                 img,
#                 "Describe this image in detail for search indexing."
#             ])
            
#             if response and response.text:
#                 return response.text.strip()
#             return f"Image from {chunk.filename} page {chunk.page_number}"
            
#         except Exception as e:
#             # Fallback text so the embedder doesn't receive an empty string
#             logger.warning(f"Image description failed: {e}. Using fallback text.")
#             return f"Image from {chunk.filename} page {chunk.page_number}"

#     @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
#     def _embed(self, text: str, task_type: str = "retrieval_document") -> Optional[list]:
#         if not text or not text.strip():
#             return None
            
#         try:
#             result = genai.embed_content(
#                 model=self.embed_model,
#                 content=text,
#                 task_type=task_type,
#             )
#             return result["embedding"]
#         except Exception as e:
#             logger.error(f"Embedding API call failed: {e}")
#             raise # Tenacity will catch this and retry



import time
import base64
import io
from typing import Optional

import google.generativeai as genai
from loguru import logger
from config import settings
from ingestion.parser import ChunkType
import PIL.Image


class GeminiEmbedder:
    def __init__(self):
        # Configure Gemini API
        genai.configure(api_key=settings.gemini_api_key, transport="rest")

        # Embedding model
        self.embed_model = settings.gemini_embed_model

        # Generation model
        # Use the model name exactly as configured in settings/.env.
        self.gen_model_name = settings.gemini_model

        # Gemini GenerativeModel for image description
        self.gen_model = genai.GenerativeModel(
            self.gen_model_name
        )

        logger.info(
            f"Embedder ready — "
            f"embed: {self.embed_model}  "
            f"gen: {self.gen_model_name}"
        )

    # ============================================================
    # EMBED DOCUMENT CHUNKS
    # ============================================================

    def embed_chunks(self, chunks: list) -> list:
        """
        Generate embeddings for all document chunks.

        Text chunks:
            → directly embedded

        Table chunks:
            → converted to text → embedded

        Image chunks:
            → described using Gemini Vision
            → description embedded

        Failed chunks are skipped instead of crashing ingestion.
        """

        results = []
        total = len(chunks)
        if total == 0:
            return results

        # --------------------------------------------------------
        # Step 1: Pre-process chunks into text representation
        # --------------------------------------------------------
        processed = []
        for i, chunk in enumerate(chunks):
            try:
                text = self._get_text(chunk)
                if not text or not text.strip():
                    text = (
                        f"Document chunk from "
                        f"{chunk.filename} "
                        f"page {chunk.page_number}"
                    )
                if chunk.chunk_type == ChunkType.IMAGE and not chunk.content:
                    chunk.content = text
                processed.append((chunk, text))
            except Exception as e:
                logger.error(f"Error preparing chunk {i + 1}: {e}")
                fallback = f"Document chunk from {chunk.filename} page {chunk.page_number}"
                processed.append((chunk, fallback))

        # --------------------------------------------------------
        # Step 2: Batch embed in chunks of 25 (15x faster than 1-by-1)
        # --------------------------------------------------------
        batch_size = 25
        num_batches = (len(processed) + batch_size - 1) // batch_size
        logger.info(f"Batch embedding {total} chunks in {num_batches} batches (batch_size={batch_size})...")

        for b_idx in range(0, len(processed), batch_size):
            batch = processed[b_idx : b_idx + batch_size]
            batch_chunks = [item[0] for item in batch]
            batch_texts = [item[1] for item in batch]
            curr_batch_num = (b_idx // batch_size) + 1

            logger.info(f"Embedding batch {curr_batch_num}/{num_batches} ({len(batch_texts)} chunks)...")

            embeddings = self._embed_batch(batch_texts, "retrieval_document")

            if embeddings and len(embeddings) == len(batch_chunks):
                for chunk, emb in zip(batch_chunks, embeddings):
                    if emb:
                        results.append((chunk, emb))
            else:
                # Safe fallback: if batch call failed, embed individually
                logger.warning(f"Batch {curr_batch_num} failed or returned partial results. Falling back to individual embedding...")
                for chunk, text in batch:
                    try:
                        emb = self._embed(text, "retrieval_document")
                        if emb:
                            results.append((chunk, emb))
                        time.sleep(0.1)
                    except Exception as e:
                        logger.error(f"Failed fallback embedding: {e}")

            # Small 0.2s pause between batches to respect rate limits
            time.sleep(0.2)

        logger.success(
            f"Successfully embedded {len(results)}/{total} chunks in batches"
        )

        return results

    # ============================================================
    # EMBED QUERY
    # ============================================================

    def embed_query(self, query: str) -> list:
        """
        Generate an embedding for a user query.
        """

        if not query or not query.strip():
            logger.warning(
                "Empty query received for embedding."
            )
            return []

        return self._embed(
            query,
            "retrieval_query"
        )

    # ============================================================
    # GET TEXT REPRESENTATION
    # ============================================================

    def _get_text(self, chunk) -> str:
        """
        Convert any chunk type into text suitable
        for the embedding model.
        """

        # --------------------------------------------------------
        # TEXT
        # --------------------------------------------------------

        if chunk.chunk_type == ChunkType.TEXT:

            return chunk.content or ""

        # --------------------------------------------------------
        # TABLE
        # --------------------------------------------------------

        elif chunk.chunk_type == ChunkType.TABLE:

            return (
                f"Table from "
                f"{chunk.filename} "
                f"page {chunk.page_number}:\n"
                f"{chunk.content or ''}"
            )

        # --------------------------------------------------------
        # IMAGE
        # --------------------------------------------------------

        elif chunk.chunk_type == ChunkType.IMAGE:

            # If image already has meaningful content,
            # don't call Gemini again.
            if (
                chunk.content
                and len(chunk.content.strip()) > 10
            ):
                return chunk.content

            # Otherwise generate an image description.
            return self._describe_image(chunk)

        # --------------------------------------------------------
        # UNKNOWN CHUNK TYPE
        # --------------------------------------------------------

        return (
            str(chunk.content)
            if chunk.content
            else ""
        )

    # ============================================================
    # IMAGE DESCRIPTION
    # ============================================================

    def _describe_image(self, chunk) -> str:
        """
        Generate a textual description of an image
        so that the description can be embedded.

        If Gemini quota is exhausted or another error occurs,
        return fallback text instead of crashing ingestion.
        """

        # --------------------------------------------------------
        # Existing content
        # --------------------------------------------------------

        if (
            chunk.content
            and len(chunk.content.strip()) > 10
        ):
            return chunk.content

        # --------------------------------------------------------
        # No image data
        # --------------------------------------------------------

        if not chunk.image_b64:

            return (
                f"Image from "
                f"{chunk.filename} "
                f"page {chunk.page_number}"
            )

        try:

            # ----------------------------------------------------
            # Decode Base64 image
            # ----------------------------------------------------

            image_bytes = base64.b64decode(
                chunk.image_b64
            )

            # ----------------------------------------------------
            # Open image using PIL
            # ----------------------------------------------------

            img = PIL.Image.open(
                io.BytesIO(image_bytes)
            )

            # ----------------------------------------------------
            # Gemini Vision request
            # ----------------------------------------------------

            response = self.gen_model.generate_content(
                [
                    img,
                    (
                        "Describe this image in detail "
                        "for semantic search indexing. "
                        "Extract important text, labels, "
                        "numbers, tables, diagrams, charts, "
                        "and meaningful visual information."
                    )
                ],
                request_options={"timeout": 15},
            )

            # ----------------------------------------------------
            # Extract response
            # ----------------------------------------------------

            if response and response.text:

                description = response.text.strip()

                if description:
                    return description

            # ----------------------------------------------------
            # Empty Gemini response
            # ----------------------------------------------------

            logger.warning(
                f"Gemini returned an empty description "
                f"for {chunk.filename} "
                f"page {chunk.page_number}"
            )

            return (
                f"Image from "
                f"{chunk.filename} "
                f"page {chunk.page_number}"
            )

        except Exception as e:

            error_message = str(e)

            # ----------------------------------------------------
            # Quota error
            # ----------------------------------------------------

            if (
                "429" in error_message
                or "quota" in error_message.lower()
                or "rate limit" in error_message.lower()
            ):

                logger.warning(
                    "Gemini API quota/rate limit "
                    "exceeded while describing image. "
                    "Using fallback text."
                )

            # ----------------------------------------------------
            # Other error
            # ----------------------------------------------------

            else:

                logger.warning(
                    f"Image description failed: "
                    f"{e}. Using fallback text."
                )

            # ----------------------------------------------------
            # Safe fallback
            # ----------------------------------------------------

            return (
                f"Image from "
                f"{chunk.filename} "
                f"page {chunk.page_number}"
            )

    # ============================================================
    # EMBEDDING API
    # ============================================================

    def _embed_batch(
        self,
        texts: list[str],
        task_type: str = "retrieval_document",
        max_retries: int = 2,
    ) -> list:
        """
        Generate Gemini embeddings in a single batch request with retry logic.
        """
        if not texts:
            return []

        sanitized = [t if (t and t.strip()) else "Document chunk" for t in texts]

        for attempt in range(max_retries + 1):
            try:
                result = genai.embed_content(
                    model=self.embed_model,
                    content=sanitized,
                    task_type=task_type,
                    output_dimensionality=768,
                    request_options={"timeout": 30},
                )
                embeddings = result.get("embedding", [])
                if embeddings and isinstance(embeddings[0], (float, int)):
                    embeddings = [embeddings]
                return embeddings
            except Exception as e:
                err_str = str(e).lower()
                if "429" in err_str or "quota" in err_str or "rate limit" in err_str:
                    logger.warning(f"Batch embedding rate limit (attempt {attempt + 1}): {e}")
                    time.sleep(1.0 * (attempt + 1))
                else:
                    logger.error(f"Batch embedding error (attempt {attempt + 1}): {e}")
                    if attempt == max_retries:
                        return []
                    time.sleep(0.5)
        return []

    def _embed(
        self,
        text: str,
        task_type: str = "retrieval_document"
    ) -> Optional[list]:
        """
        Generate Gemini embedding.

        Important:
        We intentionally DON'T use Tenacity retries here.

        If Gemini returns 429 because the quota is exhausted,
        retrying immediately only wastes time and makes the
        application appear stuck.
        """

        # --------------------------------------------------------
        # Validate text
        # --------------------------------------------------------

        if not text or not text.strip():

            logger.warning(
                "Cannot generate embedding "
                "for empty text."
            )

            return None

        try:

            # ----------------------------------------------------
            # Gemini embedding request
            # ----------------------------------------------------

            result = genai.embed_content(
                model=self.embed_model,
                content=text,
                task_type=task_type,
                output_dimensionality=768,
            )

            # ----------------------------------------------------
            # Extract embedding
            # ----------------------------------------------------

            embedding = result.get("embedding")

            if not embedding:

                logger.warning(
                    "Gemini returned an empty embedding."
                )

                return None

            return embedding

        except Exception as e:

            error_message = str(e)

            # ----------------------------------------------------
            # Gemini quota / rate limit
            # ----------------------------------------------------

            if (
                "429" in error_message
                or "quota" in error_message.lower()
                or "rate limit" in error_message.lower()
            ):

                logger.warning(
                    "Gemini embedding quota/rate limit "
                    "exceeded. Skipping this request."
                )

                return None

            # ----------------------------------------------------
            # Other Gemini error
            # ----------------------------------------------------

            logger.error(
                f"Embedding API call failed: {e}"
            )

            return None