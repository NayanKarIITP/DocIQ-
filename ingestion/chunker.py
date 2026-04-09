"""
ingestion/chunker.py
────────────────────
Splits large text chunks into smaller, overlapping windows.
Sentence-aware: never cuts mid-sentence.

Usage:
    from ingestion.chunker import TextChunker
    chunker = TextChunker(chunk_size=800, overlap=100)
    chunks = chunker.split(document_chunks)
"""

import re
from copy import deepcopy

from loguru import logger

from config import settings
from ingestion.parser import ChunkType, DocumentChunk


class TextChunker:
    """
    Splits TEXT chunks into smaller windows.
    IMAGE and TABLE chunks are passed through unchanged — they are
    already atomic units.
    """

    def __init__(
        self,
        chunk_size: int = None,
        overlap: int = None,
    ):
        self.chunk_size = chunk_size or settings.chunk_size
        self.overlap = overlap or settings.chunk_overlap

    def split(self, chunks: list[DocumentChunk]) -> list[DocumentChunk]:
        """Split text chunks; pass through images and tables."""
        result = []
        for chunk in chunks:
            if chunk.chunk_type == ChunkType.TEXT:
                result.extend(self._split_text_chunk(chunk))
            else:
                result.append(chunk)  # images/tables are atomic

        logger.info(f"Chunker: {len(chunks)} raw → {len(result)} final chunks")
        return result

    def _split_text_chunk(self, chunk: DocumentChunk) -> list[DocumentChunk]:
        """Sentence-aware sliding window split."""
        text = chunk.content.strip()
        if len(text) <= self.chunk_size:
            return [chunk]

        sentences = self._split_sentences(text)
        windows = self._build_windows(sentences)

        split_chunks = []
        for i, window_text in enumerate(windows):
            new_chunk = deepcopy(chunk)
            new_chunk.chunk_id = f"{chunk.chunk_id}-split{i}"
            new_chunk.content = window_text
            new_chunk.metadata = {
                **chunk.metadata,
                "split_index": i,
                "total_splits": len(windows),
            }
            split_chunks.append(new_chunk)

        return split_chunks

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences using regex."""
        # Split on ., !, ? followed by space+capital or end of string
        pattern = r'(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])\s*$'
        parts = re.split(pattern, text)
        # Also split on double newlines (paragraphs)
        sentences = []
        for part in parts:
            sub = [s.strip() for s in part.split('\n\n') if s.strip()]
            sentences.extend(sub)
        return sentences

    def _build_windows(self, sentences: list[str]) -> list[str]:
        """Sliding window over sentences with character-based size limit."""
        windows = []
        current_sentences = []
        current_len = 0

        for sentence in sentences:
            s_len = len(sentence)

            if current_len + s_len > self.chunk_size and current_sentences:
                # Save current window
                windows.append(" ".join(current_sentences))

                # Keep overlap: pop sentences from start until within overlap limit
                overlap_sentences = []
                overlap_len = 0
                for s in reversed(current_sentences):
                    if overlap_len + len(s) <= self.overlap:
                        overlap_sentences.insert(0, s)
                        overlap_len += len(s)
                    else:
                        break
                current_sentences = overlap_sentences
                current_len = overlap_len

            current_sentences.append(sentence)
            current_len += s_len

        # Don't forget the last window
        if current_sentences:
            windows.append(" ".join(current_sentences))

        return windows
