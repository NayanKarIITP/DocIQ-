# """retrieval/embedder.py"""
# import time, base64
# from typing import Optional
# import google.generativeai as genai
# from loguru import logger
# from tenacity import retry, stop_after_attempt, wait_exponential
# from config import settings
# from ingestion.parser import ChunkType, DocumentChunk


# class GeminiEmbedder:
#     def __init__(self):
#         genai.configure(api_key=settings.gemini_api_key)
#         self.embed_model = settings.gemini_embed_model
#         self.embed_dim   = settings.embedding_dim
#         logger.info(f"Embedder ready — embed: {self.embed_model}  gen: {settings.gemini_model}")

#     def embed_chunks(self, chunks: list) -> list:
#         results = []
#         for i, chunk in enumerate(chunks):
#             if i % 10 == 0:
#                 logger.info(f"Embedding {i}/{len(chunks)}...")
#             text = self._get_text(chunk)
#             emb  = self._embed(text, "retrieval_document")
#             if emb:
#                 if chunk.chunk_type == ChunkType.IMAGE and not chunk.content:
#                     chunk.content = text
#                 results.append((chunk, emb))
#             time.sleep(0.15)
#         logger.success(f"Embedded {len(results)}/{len(chunks)} chunks")
#         return results

#     def embed_query(self, query: str) -> list:
#         return self._embed(query, "retrieval_query")

#     def _get_text(self, chunk) -> str:
#         if chunk.chunk_type == ChunkType.TEXT:
#             return chunk.content
#         elif chunk.chunk_type == ChunkType.TABLE:
#             return f"Table from {chunk.filename} p.{chunk.page_number}:\n{chunk.content}"
#         elif chunk.chunk_type == ChunkType.IMAGE:
#             return chunk.content if chunk.content else self._describe_image(chunk)
#         return chunk.content

#     def _describe_image(self, chunk) -> str:
#         if not chunk.image_b64:
#             return f"Image from {chunk.filename} page {chunk.page_number}"
#         try:
#             import base64 as b64
#             image_bytes = b64.b64decode(chunk.image_b64)
#             # model = genai.GenerativeModel(settings.gemini_model)
#             model = genai.GenerativeModel(f"models/{settings.gemini_model}")
#             import PIL.Image, io
#             img = PIL.Image.open(io.BytesIO(image_bytes))
#             response = model.generate_content([
#                 img,
#                 "Describe this image in detail for search indexing."
#             ])
#             return response.text.strip()
#         except Exception as e:
#             logger.warning(f"Image description failed: {e}")
#             return f"Image from {chunk.filename} page {chunk.page_number}"

#     @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
#     def _embed(self, text: str, task_type: str = "retrieval_document") -> Optional[list]:
#         try:
#             result = genai.embed_content(
#                 model=self.embed_model,
#                 content=text,
#                 task_type=task_type,
#             )
#             return result["embedding"]
#         except Exception as e:
#             logger.error(f"Embedding failed: {e}")
#             raise




import time
import base64
import io
from typing import Optional
import google.generativeai as genai
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential
from config import settings
from ingestion.parser import ChunkType
import PIL.Image

class GeminiEmbedder:
    def __init__(self):
        genai.configure(api_key=settings.gemini_api_key)
        self.embed_model = settings.gemini_embed_model
        # Ensure the model name is formatted correctly for the GenerativeModel class
        self.gen_model_name = settings.gemini_model if settings.gemini_model.startswith("models/") else f"models/{settings.gemini_model}"
        self.gen_model = genai.GenerativeModel(self.gen_model_name)
        logger.info(f"Embedder ready — embed: {self.embed_model}  gen: {self.gen_model_name}")

    def embed_chunks(self, chunks: list) -> list:
        results = []
        for i, chunk in enumerate(chunks):
            if i % 10 == 0:
                logger.info(f"Embedding {i}/{len(chunks)}...")
            
            try:
                text = self._get_text(chunk)
                # If text is empty or None, provide a fallback to avoid axis 0 error
                if not text:
                    text = f"Empty chunk from {chunk.filename} p.{chunk.page_number}"
                
                emb = self._embed(text, "retrieval_document")
                
                if emb:
                    if chunk.chunk_type == ChunkType.IMAGE and not chunk.content:
                        chunk.content = text
                    results.append((chunk, emb))
            except Exception as e:
                logger.error(f"Skipping chunk {i} due to error: {e}")
                continue # This prevents the "Index out of bounds" from killing the app

            # time.sleep(0.15)
            time.sleep(1.0)  # Gemini models have a rate limit of ~300 reqs/minute, so we need to slow down the embedding loop
            
        logger.success(f"Embedded {len(results)}/{len(chunks)} chunks")
        return results

    def embed_query(self, query: str) -> list:
        return self._embed(query, "retrieval_query")

    def _get_text(self, chunk) -> str:
        if chunk.chunk_type == ChunkType.TEXT:
            return chunk.content or ""
        elif chunk.chunk_type == ChunkType.TABLE:
            return f"Table from {chunk.filename} p.{chunk.page_number}:\n{chunk.content}"
        elif chunk.chunk_type == ChunkType.IMAGE:
            return self._describe_image(chunk)
        return str(chunk.content) if chunk.content else ""

    def _describe_image(self, chunk) -> str:
        # Check if content already exists to avoid unnecessary API calls
        if chunk.content and len(chunk.content.strip()) > 10:
            return chunk.content

        if not chunk.image_b64:
            return f"Image from {chunk.filename} page {chunk.page_number}"
        
        try:
            image_bytes = base64.b64decode(chunk.image_b64)
            img = PIL.Image.open(io.BytesIO(image_bytes))
            
            response = self.gen_model.generate_content([
                img,
                "Describe this image in detail for search indexing."
            ])
            
            if response and response.text:
                return response.text.strip()
            return f"Image from {chunk.filename} page {chunk.page_number}"
            
        except Exception as e:
            # Fallback text so the embedder doesn't receive an empty string
            logger.warning(f"Image description failed: {e}. Using fallback text.")
            return f"Image from {chunk.filename} page {chunk.page_number}"

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
    def _embed(self, text: str, task_type: str = "retrieval_document") -> Optional[list]:
        if not text or not text.strip():
            return None
            
        try:
            result = genai.embed_content(
                model=self.embed_model,
                content=text,
                task_type=task_type,
            )
            return result["embedding"]
        except Exception as e:
            logger.error(f"Embedding API call failed: {e}")
            raise # Tenacity will catch this and retry