import pypdf
from pathlib import Path
from ingestion.chunker import TextChunk, chunk_document_text

class PDFParsingError(Exception):
    """Raised when a PDF file is corrupted, encrypted, or cannot be parsed."""
    pass

def extract_pdf_chunks(filepath: Path | str, doc_id: str | None = None) -> list[TextChunk]:
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    all_chunks: list[TextChunk] = []
    try:
        with open(path, "rb") as f:
            reader = pypdf.PdfReader(f)
            if reader.is_encrypted:
                raise PDFParsingError(f"PDF {path.name} is password-protected or encrypted.")
            for page_idx, page in enumerate(reader.pages):
                try:
                    page_text = page.extract_text() or ""
                except Exception as e:
                    raise PDFParsingError(f"Error extracting text from page {page_idx + 1}: {e}") from e

                if page_text.strip():
                    page_chunks = chunk_document_text(
                        text=page_text,
                        source=path.name,
                        page_number=page_idx + 1,
                        doc_id=doc_id
                    )
                    all_chunks.extend(page_chunks)
    except PDFParsingError:
        raise
    except Exception as e:
        raise PDFParsingError(f"Failed to parse PDF document {path.name}: {e}") from e

    return all_chunks

