import pypdf
from pathlib import Path
from ingestion.chunker import TextChunk, chunk_document_text

def extract_pdf_chunks(filepath: Path | str, doc_id: str | None = None) -> list[TextChunk]:
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    all_chunks: list[TextChunk] = []
    with open(path, "rb") as f:
        reader = pypdf.PdfReader(f)
        for page_idx, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""
            if page_text.strip():
                page_chunks = chunk_document_text(
                    text=page_text,
                    source=path.name,
                    page_number=page_idx + 1,
                    doc_id=doc_id
                )
                all_chunks.extend(page_chunks)
    return all_chunks
