import pytest
from ingestion.chunker import chunk_document_text, ChunkMetadata

def test_chunking_preserves_metadata_and_boundaries():
    sample_text = (
        "Daniyal is an AI Engineer specializing in advanced RAG pipelines. " * 30
        + "\n\n"
        + "He built multi-agent systems using LangChain and FastAPI. " * 30
    )
    chunks = chunk_document_text(
        text=sample_text,
        source="Daniyal_Resume.pdf",
        page_number=1,
        chunk_size=300,
        chunk_overlap=50
    )

    assert len(chunks) >= 2
    for i, chunk in enumerate(chunks):
        assert chunk.metadata.source == "Daniyal_Resume.pdf"
        assert chunk.metadata.page_number == 1
        assert chunk.metadata.chunk_index == i
        assert len(chunk.content) > 0
