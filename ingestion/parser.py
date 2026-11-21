"""
ingestion/parser.py
───────────────────
Parses PDFs into typed chunks: Text, Image, Table.
Each chunk carries rich metadata for downstream filtering.

Usage:
    from ingestion.parser import DocumentParser
    parser = DocumentParser()
    chunks = parser.parse("report.pdf")
"""

import base64
import hashlib
import io
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF
from loguru import logger
from PIL import Image
from config import settings

try:
    import camelot
    CAMELOT_AVAILABLE = True
except ImportError:
    CAMELOT_AVAILABLE = False
    logger.warning("camelot not available — table extraction will be skipped")


class ChunkType(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    TABLE = "table"


@dataclass
class DocumentChunk:
    """A single chunk extracted from a document."""
    chunk_id: str
    doc_id: str
    filename: str
    page_number: int
    chunk_type: ChunkType
    content: str                        # text content or image description
    image_b64: Optional[str] = None     # base64 image (for Image chunks)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "filename": self.filename,
            "page_number": self.page_number,
            "chunk_type": self.chunk_type.value,
            "content": self.content,
            "image_b64": self.image_b64,
            "metadata": self.metadata,
        }


class DocumentParser:
    """
    Multimodal document parser.

    Strategy:
    - Text  → extracted directly via PyMuPDF
    - Images → extracted as base64 PNG, described later by Gemini
    - Tables → extracted via camelot, converted to markdown
    """

    def __init__(self, min_text_length: int = 50, min_image_size: int = 100):
        self.min_text_length = min_text_length
        self.min_image_size = min_image_size  # pixels, filter tiny images
        try:
            fitz.TOOLS.mupdf_display_errors(False)
        except Exception:
            pass

    def parse(self, file_path: str | Path, filename: Optional[str] = None) -> list[DocumentChunk]:
        """Parse a PDF file and return all chunks."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        doc_name = filename or path.name
        doc_id = self._make_doc_id(path)
        logger.info(f"Parsing {doc_name} (doc_id={doc_id[:8]}...)")

        chunks: list[DocumentChunk] = []

        # Open with PyMuPDF
        pdf = fitz.open(str(path))
        num_pages = len(pdf)

        for page_num in range(num_pages):
            page = pdf[page_num]

            # 1. Extract text blocks
            text_chunks = self._extract_text(page, page_num, doc_id, doc_name)
            chunks.extend(text_chunks)

            # 2. Extract images (limit to top 5 images to prevent upload timeouts)
            if sum(1 for c in chunks if c.chunk_type == ChunkType.IMAGE) < 5:
                image_chunks = self._extract_images(pdf, page, page_num, doc_id, doc_name)
                chunks.extend(image_chunks)

            # 3. Fast native PyMuPDF table extraction (limit to top 5 tables to prevent RAM spikes)
            if sum(1 for c in chunks if c.chunk_type == ChunkType.TABLE) < 5:
                table_chunks = self._extract_tables_fitz(page, page_num, doc_id, doc_name)
                chunks.extend(table_chunks)

        pdf.close()

        # 4. Optional Camelot fallback only for small documents (<= 5 pages) in local dev
        has_tables = any(c.chunk_type == ChunkType.TABLE for c in chunks)
        if not has_tables and CAMELOT_AVAILABLE and num_pages <= 5 and getattr(settings, "app_env", "") != "production":
            try:
                camelot_tables = self._extract_tables(path, doc_id, doc_name)
                chunks.extend(camelot_tables)
            except Exception as e:
                logger.warning(f"Camelot fallback skipped: {e}")

        logger.success(
            f"Parsed {doc_name}: "
            f"{sum(1 for c in chunks if c.chunk_type == ChunkType.TEXT)} text, "
            f"{sum(1 for c in chunks if c.chunk_type == ChunkType.IMAGE)} image, "
            f"{sum(1 for c in chunks if c.chunk_type == ChunkType.TABLE)} table chunks"
        )
        return chunks

    def parse_bytes(self, file_bytes: bytes, filename: str) -> list[DocumentChunk]:
        """Parse PDF from bytes (for FastAPI uploads)."""
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        try:
            return self.parse(tmp_path, filename=filename)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    # ── Private helpers ───────────────────────────────────────────────────

    def _extract_text(
        self, page: fitz.Page, page_num: int, doc_id: str, filename: str
    ) -> list[DocumentChunk]:
        """Extract text blocks from a page."""
        chunks = []
        blocks = page.get_text("blocks")  # returns list of (x0,y0,x1,y1,text,...)

        for i, block in enumerate(blocks):
            text = block[4].strip()
            if len(text) < self.min_text_length:
                continue

            chunk_id = self._make_chunk_id(doc_id, page_num, "text", i)
            chunks.append(DocumentChunk(
                chunk_id=chunk_id,
                doc_id=doc_id,
                filename=filename,
                page_number=page_num + 1,
                chunk_type=ChunkType.TEXT,
                content=text,
                metadata={
                    "bbox": block[:4],
                    "block_index": i,
                },
            ))
        return chunks

    def _extract_images(
        self, pdf: fitz.Document, page: fitz.Page,
        page_num: int, doc_id: str, filename: str
    ) -> list[DocumentChunk]:
        """Extract images from a page as base64 PNG."""
        chunks = []
        image_list = page.get_images(full=True)

        for img_index, img_info in enumerate(image_list):
            xref = img_info[0]
            try:
                base_image = pdf.extract_image(xref)
                image_bytes = base_image["image"]
                img = Image.open(io.BytesIO(image_bytes))

                # Skip tiny images (icons, decorations)
                if img.width < self.min_image_size or img.height < self.min_image_size:
                    continue

                # Resize large images to conserve memory (Render 512MB limit)
                if img.width > 800 or img.height > 800:
                    img.thumbnail((800, 800))

                # Convert to compact JPEG base64
                if img.mode in ("RGBA", "P"):
                    img = img.convert("RGB")
                buffer = io.BytesIO()
                img.save(buffer, format="JPEG", quality=85)
                image_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

                chunk_id = self._make_chunk_id(doc_id, page_num, "image", img_index)
                chunks.append(DocumentChunk(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    filename=filename,
                    page_number=page_num + 1,
                    chunk_type=ChunkType.IMAGE,
                    content="",  # will be filled by Gemini Vision in embedder
                    image_b64=image_b64,
                    metadata={
                        "width": img.width,
                        "height": img.height,
                        "format": base_image.get("ext", "png"),
                    },
                ))
            except Exception as e:
                logger.warning(f"Failed to extract image {img_index} on page {page_num}: {e}")

        return chunks

    def _extract_tables_fitz(
        self, page: fitz.Page, page_num: int, doc_id: str, filename: str
    ) -> list[DocumentChunk]:
        """Fast native PyMuPDF table extraction."""
        chunks = []
        if not hasattr(page, "find_tables"):
            return chunks
        try:
            tabs = page.find_tables()
            for i, tab in enumerate(tabs):
                rows = tab.extract()
                if not rows or len(rows) < 2:
                    continue
                header = " | ".join(str(cell or "").replace("\n", " ").strip() for cell in rows[0])
                sep = " | ".join("---" for _ in rows[0])
                table_lines = [f"| {header} |", f"| {sep} |"]
                for row in rows[1:]:
                    line = " | ".join(str(cell or "").replace("\n", " ").strip() for cell in row)
                    table_lines.append(f"| {line} |")
                markdown = "\n".join(table_lines)
                chunk_id = self._make_chunk_id(doc_id, page_num, "table", i)
                chunks.append(DocumentChunk(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    filename=filename,
                    page_number=page_num + 1,
                    chunk_type=ChunkType.TABLE,
                    content=f"[TABLE]\n{markdown}",
                    metadata={
                        "rows": len(rows),
                        "cols": len(rows[0]) if rows else 0,
                    },
                ))
        except Exception as e:
            logger.debug(f"Native table extraction skipped on page {page_num}: {e}")
        return chunks

    def _extract_tables(self, path: Path, doc_id: str, filename: Optional[str] = None) -> list[DocumentChunk]:
        """Extract tables using camelot and convert to markdown."""
        chunks = []
        try:
            tables = camelot.read_pdf(str(path), pages="all", flavor="lattice")
            if not tables:
                tables = camelot.read_pdf(str(path), pages="all", flavor="stream")

            for i, table in enumerate(tables):
                df = table.df
                if df.empty:
                    continue

                # Convert to markdown table
                markdown = df.to_markdown(index=False)
                chunk_id = self._make_chunk_id(doc_id, table.page, "table", i)
                chunks.append(DocumentChunk(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    filename=filename or path.name,
                    page_number=table.page,
                    chunk_type=ChunkType.TABLE,
                    content=f"[TABLE]\n{markdown}",
                    metadata={
                        "rows": df.shape[0],
                        "cols": df.shape[1],
                        "accuracy": table.accuracy,
                    },
                ))
        except Exception as e:
            logger.warning(f"Table extraction failed: {e}")
        return chunks

    @staticmethod
    def _make_doc_id(path: Path) -> str:
        content = path.read_bytes()
        return hashlib.sha256(content).hexdigest()

    @staticmethod
    def _make_chunk_id(doc_id: str, page: int, kind: str, index: int) -> str:
        raw = f"{doc_id}-p{page}-{kind}-{index}"
        return hashlib.md5(raw.encode()).hexdigest()
