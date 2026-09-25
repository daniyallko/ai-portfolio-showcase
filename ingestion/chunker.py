from dataclasses import dataclass
from pydantic import BaseModel
from langchain_text_splitters import RecursiveCharacterTextSplitter

class ChunkMetadata(BaseModel):
    source: str
    page_number: int
    chunk_index: int
    doc_id: str | None = None

@dataclass
class TextChunk:
    content: str
    metadata: ChunkMetadata

def chunk_document_text(
    text: str,
    source: str,
    page_number: int = 1,
    chunk_size: int = 600,
    chunk_overlap: int = 100,
    doc_id: str | None = None
) -> list[TextChunk]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    raw_chunks = splitter.split_text(text)

    result = []
    for idx, content in enumerate(raw_chunks):
        meta = ChunkMetadata(
            source=source,
            page_number=page_number,
            chunk_index=idx,
            doc_id=doc_id
        )
        result.append(TextChunk(content=content.strip(), metadata=meta))
    return result
