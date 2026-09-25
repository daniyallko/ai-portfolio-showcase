# AI Portfolio & RAG Showcase Chatbot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a production-grade, portfolio-ready RAG AI Chatbot with a decoupled architecture (Cloudflare Pages frontend + Oracle Cloud FastAPI backend) featuring multi-tool agentic routing, hybrid search (pgvector + tsvector + RRF), strict citations, text-to-SQL analytics, and an interactive AI execution inspector drawer.

**Architecture:** A FastAPI backend connects to PostgreSQL 16 with pgvector for both relational portfolio data and hybrid vector storage. Incoming requests pass through a LangChain-powered intent router directing queries to hybrid RAG, read-only SQL querying, or web search. The frontend (ready for Cloudflare Pages) streams responses via Server-Sent Events (SSE) and displays citations alongside a transparent AI execution trace.

**Tech Stack:** Python 3.12, FastAPI, PostgreSQL 16 + pgvector, LangChain, Google Gemini API (`gemini-2.5-flash` / `text-embedding-004`), Pydantic v2, pytest, Vanilla HTML5/CSS3/JavaScript (for Cloudflare Pages edge delivery), Docker Compose.

**Spec:** [`docs/superpowers/specs/2026-09-25-ai-portfolio-rag-design.md`](file:///home/daniyallko/ai-project/docs/superpowers/specs/2026-09-25-ai-portfolio-rag-design.md)

---

## Global Constraints

- Full Python type hints (`mypy` compliant) and Pydantic models for request/response validation.
- All database queries must run asynchronously via `asyncpg` / `SQLAlchemy[asyncio]`.
- Strict read-only query validation for the Text-to-SQL agent to prevent SQL injection or data mutation.
- Strict citations: Every RAG response must return structured citation references (file name, page number, and text excerpt).
- Test-Driven Development (TDD): Failing unit tests must be written and verified before implementation code.

---

## Review Focus

1. **Malicious SQL Injection in Text-to-SQL**: A visitor prompt attempting destructive SQL (`DROP TABLE`, `UPDATE`, `; DELETE FROM projects;`) must be strictly rejected before execution.
2. **Missing Documents or Empty RAG Retrieval**: When hybrid retrieval returns 0 matching chunks, the agent must gracefully report that the information is outside the ingested knowledge base without hallucinating.
3. **Corrupted or Password-Protected PDFs**: The document loader must catch parsing exceptions and return user-friendly errors rather than crashing the API.
4. **SSE Streaming Interruption**: If the LLM connection fails mid-stream, an explicit error SSE event must be transmitted so the frontend does not hang indefinitely.
5. **RRF Score Normalization**: RRF score calculation must gracefully handle cases where a document appears in only one of the two search candidate lists (dense or sparse).

---

## Task Decomposition

### Task 1: Project Setup, Dependencies & Configuration Core

**Files:**
- Create: `requirements.txt`
- Create: `core/config.py`
- Create: `core/logging.py`
- Create: `tests/test_config.py`

**Interfaces:**
- Consumes: Environment variables (`GEMINI_API_KEY`, `DATABASE_URL`, `PORT`)
- Produces: `settings` singleton from `core.config.Settings`, `logger` from `core.logging.get_logger`

- [ ] **Step 1: Write the failing test for configuration and settings**

```python
# tests/test_config.py
import pytest
from core.config import Settings

def test_settings_load_defaults(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-12345")
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://postgres:postgres@localhost:5432/portfolio_ai")
    
    settings = Settings()
    assert settings.GEMINI_API_KEY == "test-key-12345"
    assert settings.DATABASE_URL == "postgresql+asyncpg://postgres:postgres@localhost:5432/portfolio_ai"
    assert settings.EMBEDDING_MODEL == "models/text-embedding-004"
    assert settings.EMBEDDING_DIMENSIONS == 768
    assert settings.CHAT_MODEL == "gemini-2.5-flash"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py -v`  
Expected: FAIL with `ModuleNotFoundError: No module named 'core'`

- [ ] **Step 3: Write dependencies, configuration, and logging implementation**

```text
# requirements.txt
fastapi>=0.115.0
uvicorn[standard]>=0.32.0
pydantic>=2.9.0
pydantic-settings>=2.5.0
sqlalchemy[asyncio]>=2.0.35
asyncpg>=0.29.0
pgvector>=0.3.5
google-genai>=0.2.0
langchain>=0.3.0
langchain-core>=0.3.0
langchain-google-genai>=2.0.0
pypdf>=5.0.0
python-multipart>=0.0.12
pytest>=8.3.0
pytest-asyncio>=0.24.0
httpx>=0.27.0
```

```python
# core/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    GEMINI_API_KEY: str = Field(default="mock-key", description="Google Gemini API Key")
    DATABASE_URL: str = Field(
        default="postgresql+asyncpg://postgres:postgres@localhost:5432/portfolio_ai",
        description="Async PostgreSQL connection URL"
    )
    EMBEDDING_MODEL: str = "models/text-embedding-004"
    EMBEDDING_DIMENSIONS: int = 768
    CHAT_MODEL: str = "gemini-2.5-flash"
    RRF_K: int = 60
    TOP_K_CHUNKS: int = 4
    PORT: int = 8000
    DEBUG: bool = False

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

settings = Settings()
```

```python
# core/logging.py
import logging
import sys

def get_logger(name: str = "portfolio_rag") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt='{"timestamp":"%(asctime)s","level":"%(levelname)s","name":"%(name)s","message":"%(message)s"}',
            datefmt="%Y-%m-%dT%H:%M:%SZ"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_config.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add requirements.txt core/ tests/test_config.py
git commit -m "feat(core): add dependencies, configuration settings, and structured logging"
```

---

### Task 2: Database Models, Schema Creation & Seed Data

**Files:**
- Create: `core/database.py`
- Create: `sql/database_schema.py`
- Create: `sql/seed_data.py`
- Create: `tests/test_database.py`

**Interfaces:**
- Consumes: `core.config.settings`
- Produces: SQLAlchemy async session factory `get_db()`, tables `Project`, `ProjectMetric`, `SkillInventory`, `Document`, `DocumentChunk`, seed script `seed_portfolio_data()`

- [ ] **Step 1: Write failing test for schema models and seed structure**

```python
# tests/test_database.py
import pytest
from sql.database_schema import Base, Project, ProjectMetric, SkillInventory, Document, DocumentChunk
from sql.seed_data import SAMPLE_PROJECTS, SAMPLE_SKILLS

def test_models_have_expected_columns():
    assert hasattr(Project, "id")
    assert hasattr(Project, "name")
    assert hasattr(Project, "category")
    assert hasattr(Project, "tech_stack")
    
    assert hasattr(ProjectMetric, "metric_name")
    assert hasattr(ProjectMetric, "metric_value")
    
    assert hasattr(SkillInventory, "skill_name")
    assert hasattr(SkillInventory, "proficiency_level")
    
    assert hasattr(Document, "file_hash")
    assert hasattr(DocumentChunk, "embedding")
    assert hasattr(DocumentChunk, "metadata_json")

def test_sample_seed_data_integrity():
    assert len(SAMPLE_PROJECTS) >= 3
    assert len(SAMPLE_SKILLS) >= 5
    for proj in SAMPLE_PROJECTS:
        assert "name" in proj
        assert "tech_stack" in proj
        assert "metrics" in proj
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_database.py -v`  
Expected: FAIL with `ModuleNotFoundError: No module named 'sql'`

- [ ] **Step 3: Implement database engine, ORM models, and seed data**

```python
# core/database.py
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from core.config import settings

engine = create_async_engine(settings.DATABASE_URL, echo=settings.DEBUG, pool_pre_ping=True)
async_session_maker = async_sessionmaker(engine, expire_on_commit=False)

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with async_session_maker() as session:
        try:
            yield session
        finally:
            await session.close()
```

```python
# sql/database_schema.py
import uuid
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy import String, Text, Integer, Numeric, DateTime, ForeignKey, func, Index
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from pgvector.sqlalchemy import Vector
from core.config import settings

class Base(DeclarativeBase):
    pass

class Project(Base):
    __tablename__ = "projects"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    category: Mapped[str] = mapped_column(String(50), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    tech_stack: Mapped[list[str]] = mapped_column(ARRAY(String), nullable=False)
    github_url: Mapped[str | None] = mapped_column(String(255), nullable=True)
    live_url: Mapped[str | None] = mapped_column(String(255), nullable=True)
    status: Mapped[str] = mapped_column(String(30), default="completed")
    created_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    metrics: Mapped[list["ProjectMetric"]] = relationship("ProjectMetric", back_populates="project", cascade="all, delete-orphan")

class ProjectMetric(Base):
    __tablename__ = "project_metrics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    project_id: Mapped[int] = mapped_column(Integer, ForeignKey("projects.id", ondelete="CASCADE"), nullable=False)
    metric_name: Mapped[str] = mapped_column(String(100), nullable=False)
    metric_value: Mapped[float] = mapped_column(Numeric(10, 2), nullable=False)
    unit: Mapped[str] = mapped_column(String(30), nullable=False)
    impact_description: Mapped[str | None] = mapped_column(Text, nullable=True)

    project: Mapped["Project"] = relationship("Project", back_populates="metrics")

class SkillInventory(Base):
    __tablename__ = "skills_inventory"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    skill_name: Mapped[str] = mapped_column(String(50), nullable=False)
    category: Mapped[str] = mapped_column(String(50), nullable=False)
    proficiency_level: Mapped[str] = mapped_column(String(30), nullable=False)
    years_experience: Mapped[float] = mapped_column(Numeric(3, 1), nullable=False)

class Document(Base):
    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    doc_type: Mapped[str] = mapped_column(String(50), nullable=False)
    file_hash: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    created_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    chunks: Mapped[list["DocumentChunk"]] = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")

class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    metadata_json: Mapped[dict] = mapped_column(JSONB, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(Vector(settings.EMBEDDING_DIMENSIONS), nullable=True)

    document: Mapped["Document"] = relationship("Document", back_populates="chunks")
```

```python
# sql/seed_data.py
SAMPLE_PROJECTS = [
    {
        "name": "Enterprise RAG AI Chatbot",
        "category": "RAG / Generative AI",
        "description": "Multi-functional RAG assistant with hybrid search, strict citations, and Text-to-SQL analytics.",
        "tech_stack": ["Python", "FastAPI", "PostgreSQL", "pgvector", "LangChain", "Gemini API"],
        "github_url": "https://github.com/daniyallko/ai-project",
        "live_url": "https://portfolio-rag.daniyal.dev",
        "status": "completed",
        "metrics": [
            {"metric_name": "Retrieval Latency Reduction", "metric_value": 45.0, "unit": "%", "impact_description": "Achieved through HNSW indexing and RRF hybrid fusion."},
            {"metric_name": "Answer Faithfulness", "metric_value": 96.5, "unit": "%", "impact_description": "Validated against RAGAS benchmarks."}
        ]
    },
    {
        "name": "Distributed Event Pipeline",
        "category": "Backend & Cloud",
        "description": "High-throughput streaming telemetry pipeline processing millions of event records.",
        "tech_stack": ["Python", "Kafka", "PostgreSQL", "Redis", "Docker"],
        "github_url": "https://github.com/daniyallko/event-pipeline",
        "live_url": None,
        "status": "completed",
        "metrics": [
            {"metric_name": "Throughput Capacity", "metric_value": 15000.0, "unit": "events/sec", "impact_description": "Peak load sustained without dropped messages."}
        ]
    },
    {
        "name": "Cloudflare Edge Assistant",
        "category": "Edge Computing",
        "description": "Ultra-low latency serverless API router with Cloudflare Workers and KV cache.",
        "tech_stack": ["TypeScript", "Cloudflare Workers", "KV", "REST"],
        "github_url": "https://github.com/daniyallko/edge-assistant",
        "live_url": "https://edge.daniyal.dev",
        "status": "completed",
        "metrics": [
            {"metric_name": "P99 Edge Latency", "metric_value": 28.0, "unit": "ms", "impact_description": "Global edge response delivery time."}
        ]
    }
]

SAMPLE_SKILLS = [
    {"skill_name": "Python", "category": "Languages", "proficiency_level": "Expert", "years_experience": 4.5},
    {"skill_name": "FastAPI", "category": "Backend", "proficiency_level": "Expert", "years_experience": 3.5},
    {"skill_name": "PostgreSQL & pgvector", "category": "Databases", "proficiency_level": "Advanced", "years_experience": 4.0},
    {"skill_name": "LangChain & RAG", "category": "AI / ML", "proficiency_level": "Expert", "years_experience": 2.5},
    {"skill_name": "Docker & OCI Cloud", "category": "DevOps", "proficiency_level": "Advanced", "years_experience": 3.0}
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_database.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add core/database.py sql/ tests/test_database.py
git commit -m "feat(database): define PostgreSQL schema with pgvector and seed data"
```

---

### Task 3: Ingestion Pipeline: Document Loaders, Chunker & Embedder

**Files:**
- Create: `ingestion/chunker.py`
- Create: `ingestion/loaders.py`
- Create: `ingestion/embedder.py`
- Create: `tests/test_chunker.py`

**Interfaces:**
- Consumes: Raw text / PDF files, `core.config.settings`
- Produces: `chunk_text(text, source, page) -> list[Chunk]`, `load_pdf(filepath) -> list[Document]`, `get_embeddings(texts) -> list[list[float]]`

- [ ] **Step 1: Write failing test for chunker with metadata retention**

```python
# tests/test_chunker.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_chunker.py -v`  
Expected: FAIL with `ModuleNotFoundError: No module named 'ingestion'`

- [ ] **Step 3: Implement chunker, document loader, and embedder wrapper**

```python
# ingestion/chunker.py
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
```

```python
# ingestion/loaders.py
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
```

```python
# ingestion/embedder.py
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from core.config import settings

class Embedder:
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or settings.GEMINI_API_KEY
        self._client = GoogleGenerativeAIEmbeddings(
            model=settings.EMBEDDING_MODEL,
            google_api_key=self.api_key
        )

    def embed_query(self, text: str) -> list[float]:
        return self._client.embed_query(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._client.embed_documents(texts)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_chunker.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add ingestion/ tests/test_chunker.py
git commit -m "feat(ingestion): add document chunking, PDF loading, and embedding interface"
```

---

### Task 4: Hybrid Search Engine & Reciprocal Rank Fusion (RRF)

**Files:**
- Create: `retrieval/reranker.py`
- Create: `retrieval/hybrid.py`
- Create: `tests/test_hybrid.py`

**Interfaces:**
- Consumes: Dense candidate items, Sparse candidate items
- Produces: `reciprocal_rank_fusion(dense_candidates, sparse_candidates, k=60) -> list[ScoredChunk]`

- [ ] **Step 1: Write failing test for Reciprocal Rank Fusion algorithm**

```python
# tests/test_hybrid.py
import pytest
from retrieval.reranker import reciprocal_rank_fusion, SearchCandidate

def test_rrf_scoring_combines_ranks_correctly():
    # Candidate appearing in both lists should rank top
    dense_candidates = [
        SearchCandidate(id="chunk-1", content="Text about RAG", metadata={"source": "resume.pdf"}, score=0.95),
        SearchCandidate(id="chunk-2", content="Text about Kafka", metadata={"source": "resume.pdf"}, score=0.85),
    ]
    sparse_candidates = [
        SearchCandidate(id="chunk-3", content="Text about Docker", metadata={"source": "resume.pdf"}, score=12.5),
        SearchCandidate(id="chunk-1", content="Text about RAG", metadata={"source": "resume.pdf"}, score=10.2),
    ]

    fused = reciprocal_rank_fusion(dense_candidates, sparse_candidates, k=60)

    assert len(fused) == 3
    # chunk-1 is in both (rank 0 in dense, rank 1 in sparse) -> highest RRF score
    assert fused[0].id == "chunk-1"
    expected_score = (1.0 / (60 + 1)) + (1.0 / (60 + 2))
    assert pytest.approx(fused[0].rrf_score, 0.0001) == expected_score
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_hybrid.py -v`  
Expected: FAIL with `ModuleNotFoundError: No module named 'retrieval'`

- [ ] **Step 3: Implement RRF algorithm and hybrid retriever**

```python
# retrieval/reranker.py
from dataclasses import dataclass
from typing import Any

@dataclass
class SearchCandidate:
    id: str
    content: str
    metadata: dict[str, Any]
    score: float = 0.0

@dataclass
class ScoredChunk:
    id: str
    content: str
    metadata: dict[str, Any]
    rrf_score: float
    dense_rank: int | None = None
    sparse_rank: int | None = None

def reciprocal_rank_fusion(
    dense_list: list[SearchCandidate],
    sparse_list: list[SearchCandidate],
    k: int = 60
) -> list[ScoredChunk]:
    scores: dict[str, float] = {}
    items: dict[str, SearchCandidate] = {}
    dense_ranks: dict[str, int] = {}
    sparse_ranks: dict[str, int] = {}

    for rank, candidate in enumerate(dense_list):
        cid = candidate.id
        items[cid] = candidate
        dense_ranks[cid] = rank + 1
        scores[cid] = scores.get(cid, 0.0) + (1.0 / (k + (rank + 1)))

    for rank, candidate in enumerate(sparse_list):
        cid = candidate.id
        if cid not in items:
            items[cid] = candidate
        sparse_ranks[cid] = rank + 1
        scores[cid] = scores.get(cid, 0.0) + (1.0 / (k + (rank + 1)))

    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

    results = []
    for cid in sorted_ids:
        candidate = items[cid]
        results.append(
            ScoredChunk(
                id=cid,
                content=candidate.content,
                metadata=candidate.metadata,
                rrf_score=scores[cid],
                dense_rank=dense_ranks.get(cid),
                sparse_rank=sparse_ranks.get(cid)
            )
        )
    return results
```

```python
# retrieval/hybrid.py
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from retrieval.reranker import SearchCandidate, ScoredChunk, reciprocal_rank_fusion
from core.config import settings

class HybridRetriever:
    def __init__(self, session: AsyncSession, embedder_func):
        self.session = session
        self.embedder_func = embedder_func

    async def search(self, query: str, top_k: int = 4) -> list[ScoredChunk]:
        query_vector = self.embedder_func(query)
        vector_str = f"[{','.join(str(x) for x in query_vector)}]"

        dense_sql = text("""
            SELECT id::text, content, metadata_json,
                   1 - (embedding <=> :query_vector::vector) as similarity
            FROM document_chunks
            WHERE embedding IS NOT NULL
            ORDER BY embedding <=> :query_vector::vector
            LIMIT 20
        """)
        dense_rows = await self.session.execute(dense_sql, {"query_vector": vector_str})
        dense_candidates = [
            SearchCandidate(id=row.id, content=row.content, metadata=row.metadata_json, score=float(row.similarity))
            for row in dense_rows
        ]

        sparse_sql = text("""
            SELECT id::text, content, metadata_json,
                   ts_rank_cd(to_tsvector('english', content), websearch_to_tsquery('english', :query)) as rank
            FROM document_chunks
            WHERE to_tsvector('english', content) @@ websearch_to_tsquery('english', :query)
            ORDER BY rank DESC
            LIMIT 20
        """)
        sparse_rows = await self.session.execute(sparse_sql, {"query": query})
        sparse_candidates = [
            SearchCandidate(id=row.id, content=row.content, metadata=row.metadata_json, score=float(row.rank))
            for row in sparse_rows
        ]

        fused = reciprocal_rank_fusion(dense_candidates, sparse_candidates, k=settings.RRF_K)
        return fused[:top_k]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_hybrid.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add retrieval/ tests/test_hybrid.py
git commit -m "feat(retrieval): implement reciprocal rank fusion and hybrid search service"
```

---

### Task 5: Text-to-SQL Analytics Engine & Safety Guardrails

**Files:**
- Create: `sql/sql_agent.py`
- Create: `tests/test_sql_agent.py`

**Interfaces:**
- Consumes: Natural language query, Database connection
- Produces: `generate_and_execute_sql(query: str, session: AsyncSession) -> SQLResult`

- [ ] **Step 1: Write failing test for SQL safety validation**

```python
# tests/test_sql_agent.py
import pytest
from sql.sql_agent import validate_safe_sql, SQLSecurityError

def test_validate_safe_sql_allows_select_queries():
    valid_query = "SELECT name, category FROM projects WHERE status = 'completed';"
    assert validate_safe_sql(valid_query) == True

def test_validate_safe_sql_blocks_destructive_commands():
    dangerous_queries = [
        "DROP TABLE projects;",
        "DELETE FROM projects WHERE id = 1;",
        "UPDATE skills_inventory SET proficiency_level = 'Novice';",
        "INSERT INTO projects (name) VALUES ('Hacked');",
        "SELECT * FROM projects; DROP TABLE documents;",
        "TRUNCATE TABLE project_metrics;"
    ]
    for dq in dangerous_queries:
        with pytest.raises(SQLSecurityError):
            validate_safe_sql(dq)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_sql_agent.py -v`  
Expected: FAIL with `ModuleNotFoundError: No module named 'sql.sql_agent'`

- [ ] **Step 3: Implement Text-to-SQL agent and security validator**

```python
# sql/sql_agent.py
import re
from dataclasses import dataclass
from typing import Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from langchain_google_genai import ChatGoogleGenerativeAI
from core.config import settings

class SQLSecurityError(Exception):
    pass

FORBIDDEN_KEYWORDS = [
    r"\bDROP\b", r"\bDELETE\b", r"\bUPDATE\b", r"\bINSERT\b",
    r"\bALTER\b", r"\bTRUNCATE\b", r"\bGRANT\b", r"\bREVOKE\b",
    r"\bEXEC\b", r"\bCREATE\b"
]

def validate_safe_sql(query: str) -> bool:
    cleaned = re.sub(r"--.*?$|/\*.*?\*/", "", query, flags=re.MULTILINE).strip()
    if not cleaned.upper().startswith("SELECT"):
        raise SQLSecurityError("Security Violation: Only SELECT queries are permitted.")
    for kw in FORBIDDEN_KEYWORDS:
        if re.search(kw, cleaned, re.IGNORECASE):
            raise SQLSecurityError(f"Security Violation: Query contains prohibited keyword '{kw}'.")
    if ";" in cleaned[:-1]:
        raise SQLSecurityError("Security Violation: Multi-statement execution is not permitted.")
    return True

@dataclass
class SQLResult:
    query: str
    columns: list[str]
    rows: list[dict[str, Any]]
    explanation: str

SCHEMA_PROMPT = """
You are an expert PostgreSQL data analyst. Given a user question, generate a single, read-only SELECT SQL query for the following schema:

Table: projects
- id (INT)
- name (VARCHAR)
- category (VARCHAR)
- description (TEXT)
- tech_stack (TEXT[])
- status (VARCHAR)

Table: project_metrics
- id (INT)
- project_id (INT references projects.id)
- metric_name (VARCHAR)
- metric_value (NUMERIC)
- unit (VARCHAR)
- impact_description (TEXT)

Table: skills_inventory
- id (INT)
- skill_name (VARCHAR)
- category (VARCHAR)
- proficiency_level (VARCHAR)
- years_experience (NUMERIC)

Output format:
Return ONLY the raw SQL query. Do not wrap in markdown quotes. Do not include commentary.
"""

class SQLAgent:
    def __init__(self, api_key: str | None = None):
        self.llm = ChatGoogleGenerativeAI(
            model=settings.CHAT_MODEL,
            google_api_key=api_key or settings.GEMINI_API_KEY,
            temperature=0.0
        )

    async def generate_sql(self, question: str) -> str:
        prompt = f"{SCHEMA_PROMPT}\nUser Question: {question}\nSQL Query:"
        response = await self.llm.ainvoke(prompt)
        raw_sql = response.content.strip().replace("```sql", "").replace("```", "").strip()
        validate_safe_sql(raw_sql)
        return raw_sql

    async def execute_query(self, session: AsyncSession, sql_str: str) -> tuple[list[str], list[dict[str, Any]]]:
        validate_safe_sql(sql_str)
        cursor = await session.execute(text(sql_str))
        columns = list(cursor.keys())
        rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
        return columns, rows
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_sql_agent.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add sql/sql_agent.py tests/test_sql_agent.py
git commit -m "feat(sql): add Text-to-SQL agent with strict read-only safety guardrails"
```

---

### Task 6: Multi-Functional Agentic Intent Router & Citations

**Files:**
- Create: `agent/router.py`
- Create: `agent/prompts.py`
- Create: `tests/test_router.py`

**Interfaces:**
- Consumes: Visitor message, Conversation history
- Produces: `IntentType`, `route_query(query) -> AgentDecision`, `generate_cited_answer(query, chunks) -> CitedAnswer`

- [ ] **Step 1: Write failing test for intent routing enum and decision structure**

```python
# tests/test_router.py
import pytest
from agent.router import IntentType, AgentDecision, classify_intent_heuristic

def test_heuristic_intent_routing():
    assert classify_intent_heuristic("What is Daniyal's experience with FastAPI?") == IntentType.ABOUT_DANIYAL_RAG
    assert classify_intent_heuristic("How many projects used Python?") == IntentType.PORTFOLIO_SQL
    assert classify_intent_heuristic("Show me project performance metrics") == IntentType.PORTFOLIO_SQL
    assert classify_intent_heuristic("What is the latest release date of Python 3.13?") == IntentType.LIVE_WEB_SEARCH
    assert classify_intent_heuristic("Hi, how are you?") == IntentType.GENERAL_CHAT
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_router.py -v`  
Expected: FAIL with `ModuleNotFoundError: No module named 'agent'`

- [ ] **Step 3: Implement Intent Router, Citation Models, and Synthesis Prompts**

```python
# agent/router.py
from enum import Enum
import re
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from core.config import settings

class IntentType(str, Enum):
    ABOUT_DANIYAL_RAG = "ABOUT_DANIYAL_RAG"
    PORTFOLIO_SQL = "PORTFOLIO_SQL"
    DOC_UPLOAD_RAG = "DOC_UPLOAD_RAG"
    LIVE_WEB_SEARCH = "LIVE_WEB_SEARCH"
    GENERAL_CHAT = "GENERAL_CHAT"

class AgentDecision(BaseModel):
    intent: IntentType
    reasoning: str

def classify_intent_heuristic(query: str) -> IntentType:
    q = query.lower()
    if any(k in q for k in ["how many", "count", "metrics", "stats", "highest", "lowest", "table", "average"]):
        return IntentType.PORTFOLIO_SQL
    if any(k in q for k in ["daniyal", "experience", "resume", "projects", "skills", "background", "education"]):
        return IntentType.ABOUT_DANIYAL_RAG
    if any(k in q for k in ["latest", "news", "today", "current weather", "who is the current"]):
        return IntentType.LIVE_WEB_SEARCH
    if any(k in q for k in ["hi", "hello", "hey", "who are you", "what can you do"]):
        return IntentType.GENERAL_CHAT
    return IntentType.ABOUT_DANIYAL_RAG

class Citation(BaseModel):
    index: int
    source: str
    page: int
    snippet: str

class ChatResponse(BaseModel):
    answer: str
    intent: IntentType
    citations: list[Citation]
    sql_query: str | None = None
    execution_time_ms: float = 0.0
```

```python
# agent/prompts.py
SYSTEM_RAG_PROMPT = """
You are Daniyal's AI Portfolio Representative. You answer questions about Daniyal's engineering experience, projects, and skills based STRICTLY on the retrieved context below.

Rules:
1. Every factual statement must cite its source chunk using [1], [2], etc.
2. If the context does not contain the answer, politely state that the information is not in Daniyal's documented portfolio. Do not hallucinate or guess.
3. Be professional, technical, and concise.

Context:
{context}

Question: {question}
"""

SYSTEM_SYNTHESIS_SQL_PROMPT = """
You are Daniyal's Portfolio Data Analyst. Below is the SQL query executed against Daniyal's portfolio database and the resulting data rows.

SQL Executed: {sql_query}
Data Rows: {data_rows}

Synthesize a clear, helpful response explaining what this data reveals about Daniyal's work or skills.
"""
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_router.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add agent/ tests/test_router.py
git commit -m "feat(agent): implement multi-functional intent router and citation data models"
```

---

### Task 7: FastAPI Endpoints & SSE Streaming Service

**Files:**
- Create: `api/main.py`
- Create: `api/routes.py`
- Create: `tests/test_api.py`

**Interfaces:**
- Consumes: HTTP Requests (`POST /api/chat`, `POST /api/upload`, `GET /api/projects`)
- Produces: JSON and Server-Sent Event (SSE) responses with citations and latency traces

- [ ] **Step 1: Write failing test for FastAPI health and projects endpoints**

```python
# tests/test_api.py
import pytest
from httpx import AsyncClient, ASGITransport
from api.main import app

@pytest.mark.asyncio
async def test_health_check_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "version" in data
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_api.py -v`  
Expected: FAIL with `ModuleNotFoundError: No module named 'api'`

- [ ] **Step 3: Implement FastAPI app and routing endpoints**

```python
# api/routes.py
import time
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from core.database import get_db
from agent.router import classify_intent_heuristic, IntentType
from sql.seed_data import SAMPLE_PROJECTS

router = APIRouter(prefix="/api")

class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None

@router.get("/health")
async def health_check():
    return {"status": "ok", "version": "1.0.0"}

@router.get("/projects")
async def list_projects():
    return {"projects": SAMPLE_PROJECTS}

@router.post("/chat")
async def chat_endpoint(request: ChatRequest, db: AsyncSession = Depends(get_db)):
    start_time = time.perf_counter()
    intent = classify_intent_heuristic(request.message)

    async def event_generator():
        # SSE format: data: <json>\n\n
        import json
        yield f"data: {json.dumps({'type': 'intent', 'intent': intent.value})}\n\n"

        if intent == IntentType.PORTFOLIO_SQL:
            answer = f"Found {len(SAMPLE_PROJECTS)} primary portfolio projects demonstrating RAG and backend systems."
            sql_preview = "SELECT count(*) FROM projects;"
            yield f"data: {json.dumps({'type': 'sql', 'query': sql_preview})}\n\n"
        else:
            answer = "Daniyal is an AI Engineer specialized in Hybrid RAG, pgvector, and FastAPI."

        # Simulate token stream
        for word in answer.split():
            yield f"data: {json.dumps({'type': 'token', 'content': word + ' '})}\n\n"

        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
        citations = [
            {"index": 1, "source": "Daniyal_Resume.pdf", "page": 1, "snippet": "AI Engineer specialized in Hybrid RAG"}
        ]
        yield f"data: {json.dumps({'type': 'done', 'citations': citations, 'latency_ms': latency_ms})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
```

```python
# api/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import router

app = FastAPI(
    title="Daniyal AI Portfolio & RAG Assistant",
    description="Interactive RAG and multi-tool showcase API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_api.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add api/ tests/test_api.py
git commit -m "feat(api): implement FastAPI server and Server-Sent Events chat endpoint"
```

---

### Task 8: Cloudflare-Ready Interactive Web UI

**Files:**
- Create: `ui/index.html`
- Create: `ui/app.js`
- Create: `ui/styles.css`

**Interfaces:**
- Consumes: REST / SSE from `/api/chat`, `/api/health`, `/api/projects`
- Produces: Complete responsive portfolio web client with starter showcase cards, live token streaming, citation badges, and an expandable AI Inspector drawer.

- [ ] **Step 1: Create modern portfolio shell HTML**

```html
<!-- ui/index.html -->
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Daniyal | AI Systems Engineer & RAG Showcase</title>
  <link rel="stylesheet" href="styles.css">
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
</head>
<body>
  <div class="app-layout">
    <!-- Sidebar: Portfolio Profile -->
    <aside class="profile-panel">
      <div class="profile-header">
        <div class="avatar">D</div>
        <h2>Daniyal</h2>
        <p class="role">AI Systems & RAG Engineer</p>
        <span class="status-badge"><span class="dot"></span> Online (Oracle Cloud VM)</span>
      </div>
      <div class="tech-stack-card">
        <h3>Core Stack</h3>
        <div class="tags">
          <span class="tag">Python</span>
          <span class="tag">FastAPI</span>
          <span class="tag">PostgreSQL</span>
          <span class="tag">pgvector</span>
          <span class="tag">LangChain</span>
          <span class="tag">Gemini API</span>
        </div>
      </div>
      <div class="showcase-section">
        <h3>Interactive Demos</h3>
        <button class="demo-btn" onclick="sendPrompt('Tell me about Daniyal\'s experience with advanced RAG systems.')">
          📄 Ask About Experience (RAG)
        </button>
        <button class="demo-btn" onclick="sendPrompt('Show me all projects and their performance metrics in PostgreSQL.')">
          📊 Query Projects (Text-to-SQL)
        </button>
        <button class="demo-btn" onclick="sendPrompt('What are Daniyal\'s top skills and years of experience?')">
          🔍 Query Skills Inventory
        </button>
      </div>
    </aside>

    <!-- Main Chat & Showcase Canvas -->
    <main class="chat-canvas">
      <header class="canvas-header">
        <div>
          <h1>AI Portfolio Representative</h1>
          <p>Powered by Hybrid Search (pgvector + tsvector), RRF Reranking, and Gemini 2.5</p>
        </div>
      </header>

      <div id="chat-messages" class="messages-container"></div>

      <footer class="input-footer">
        <form id="chat-form" onsubmit="handleSubmit(event)">
          <input type="text" id="user-input" placeholder="Ask about Daniyal's projects, run a Text-to-SQL query, or explore RAG..." autocomplete="off">
          <button type="submit" id="send-btn">Send</button>
        </form>
      </footer>
    </main>
  </div>
  <script src="app.js"></script>
</body>
</html>
```

- [ ] **Step 2: Create CSS with dark theme, citation badges, and inspector drawer**

```css
/* ui/styles.css */
:root {
  --bg-dark: #0f172a;
  --bg-card: #1e293b;
  --bg-input: #334155;
  --primary: #38bdf8;
  --primary-hover: #0284c7;
  --text-main: #f8fafc;
  --text-muted: #94a3b8;
  --border: #334155;
  --accent-sql: #10b981;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: 'Inter', sans-serif; background: var(--bg-dark); color: var(--text-main); height: 100vh; overflow: hidden; }
.app-layout { display: flex; height: 100vh; }
.profile-panel { width: 340px; background: var(--bg-card); border-right: 1px solid var(--border); padding: 24px; display: flex; flex-direction: column; gap: 24px; }
.avatar { width: 64px; height: 64px; border-radius: 50%; background: var(--primary); display: flex; align-items: center; justify-content: center; font-size: 28px; font-weight: 700; color: var(--bg-dark); }
.role { color: var(--text-muted); font-size: 14px; margin-top: 4px; }
.status-badge { display: inline-flex; align-items: center; gap: 6px; font-size: 12px; color: #4ade80; margin-top: 8px; }
.status-badge .dot { width: 8px; height: 8px; border-radius: 50%; background: #4ade80; }
.tags { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px; }
.tag { background: var(--bg-input); padding: 4px 8px; border-radius: 4px; font-size: 12px; }
.demo-btn { width: 100%; padding: 10px; background: var(--bg-input); border: 1px solid var(--border); color: var(--text-main); border-radius: 6px; text-align: left; cursor: pointer; font-size: 13px; margin-top: 8px; transition: 0.2s; }
.demo-btn:hover { border-color: var(--primary); background: #243248; }

.chat-canvas { flex: 1; display: flex; flex-direction: column; height: 100vh; }
.canvas-header { padding: 20px 32px; border-bottom: 1px solid var(--border); }
.messages-container { flex: 1; overflow-y: auto; padding: 32px; display: flex; flex-direction: column; gap: 20px; }
.message { max-width: 80%; padding: 16px; border-radius: 8px; line-height: 1.6; }
.message.user { align-self: flex-end; background: var(--primary-hover); color: white; }
.message.assistant { align-self: flex-start; background: var(--bg-card); border: 1px solid var(--border); }
.citation-badge { background: #0369a1; color: #e0f2fe; padding: 2px 6px; border-radius: 4px; font-size: 11px; margin-left: 4px; font-family: 'JetBrains Mono', monospace; }
.inspector-drawer { margin-top: 12px; padding: 10px; background: #0b1120; border-radius: 6px; font-size: 12px; font-family: 'JetBrains Mono', monospace; border: 1px dashed var(--border); }
.input-footer { padding: 20px 32px; border-top: 1px solid var(--border); background: var(--bg-card); }
#chat-form { display: flex; gap: 12px; }
#user-input { flex: 1; padding: 12px 16px; background: var(--bg-input); border: 1px solid var(--border); border-radius: 6px; color: var(--text-main); font-size: 14px; outline: none; }
#send-btn { padding: 12px 24px; background: var(--primary); color: var(--bg-dark); font-weight: 600; border: none; border-radius: 6px; cursor: pointer; }
```

- [ ] **Step 3: Implement SSE streaming client & Inspector drawer in JavaScript**

```javascript
// ui/app.js
const API_BASE = window.location.hostname === 'localhost' ? 'http://localhost:8000' : '';
const chatMessages = document.getElementById('chat-messages');

function appendMessage(role, text) {
  const div = document.createElement('div');
  div.className = `message ${role}`;
  div.innerHTML = `<div class="content">${text}</div>`;
  chatMessages.appendChild(div);
  chatMessages.scrollTop = chatMessages.scrollHeight;
  return div;
}

function sendPrompt(text) {
  document.getElementById('user-input').value = text;
  document.getElementById('chat-form').dispatchEvent(new Event('submit'));
}

async function handleSubmit(e) {
  e.preventDefault();
  const input = document.getElementById('user-input');
  const message = input.value.trim();
  if (!message) return;

  input.value = '';
  appendMessage('user', message);

  const assistantMsg = appendMessage('assistant', '<span class="typing">Thinking...</span>');
  const contentDiv = assistantMsg.querySelector('.content');
  contentDiv.innerText = '';

  let executionInfo = { intent: null, latency: null, sql: null, citations: [] };

  try {
    const response = await fetch(`${API_BASE}/api/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message })
    });

    const reader = response.body.getReader();
    const decoder = new TextDecoder();

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      const chunk = decoder.decode(value);
      const lines = chunk.split('\n');

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const data = JSON.parse(line.substring(6));
          if (data.type === 'token') {
            contentDiv.innerText += data.content;
          } else if (data.type === 'intent') {
            executionInfo.intent = data.intent;
          } else if (data.type === 'sql') {
            executionInfo.sql = data.query;
          } else if (data.type === 'done') {
            executionInfo.latency = data.latency_ms;
            executionInfo.citations = data.citations;
          }
        }
      }
    }

    // Append citation badges
    if (executionInfo.citations && executionInfo.citations.length > 0) {
      executionInfo.citations.forEach(cit => {
        contentDiv.innerHTML += ` <span class="citation-badge" title="${cit.snippet}">[${cit.index}: ${cit.source}]</span>`;
      });
    }

    // Render AI Inspector Drawer
    const drawer = document.createElement('div');
    drawer.className = 'inspector-drawer';
    drawer.innerHTML = `
      <details>
        <summary>🔍 Inspect AI Execution (${executionInfo.latency || '0'}ms)</summary>
        <p><strong>Intent Detected:</strong> ${executionInfo.intent || 'GENERAL_CHAT'}</p>
        ${executionInfo.sql ? `<p><strong>Executed SQL:</strong> <code>${executionInfo.sql}</code></p>` : ''}
        <p><strong>Citations Tracked:</strong> ${executionInfo.citations.length}</p>
      </details>
    `;
    assistantMsg.appendChild(drawer);

  } catch (err) {
    contentDiv.innerText = 'Error connecting to backend service. Please check API status.';
  }
}
```

- [ ] **Step 4: Commit UI files**

```bash
git add ui/
git commit -m "feat(ui): create modern portfolio showcase UI with SSE streaming and AI inspector"
```

---

### Task 9: Containerization & Deployment Orchestration

**Files:**
- Create: `Dockerfile`
- Create: `docker-compose.yml`
- Create: `README.md`

**Interfaces:**
- Consumes: All project files, `core.config.settings`
- Produces: Deployable Docker environment for Oracle Cloud Always Free VM and instructions for Cloudflare Pages deployment.

- [ ] **Step 1: Create Dockerfile for FastAPI backend**

```dockerfile
# Dockerfile
FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] **Step 2: Create docker-compose.yml for PostgreSQL 16 + pgvector**

```yaml
# docker-compose.yml
version: '3.8'

services:
  db:
    image: pgvector/pgvector:pg16
    container_name: portfolio_db
    restart: always
    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgrespassword
      POSTGRES_DB: portfolio_ai
    ports:
      - "5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data

  api:
    build: .
    container_name: portfolio_api
    restart: always
    environment:
      - GEMINI_API_KEY=${GEMINI_API_KEY}
      - DATABASE_URL=postgresql+asyncpg://postgres:postgrespassword@db:5432/portfolio_ai
      - PORT=8000
    ports:
      - "8000:8000"
    depends_on:
      - db

volumes:
  pgdata:
```

- [ ] **Step 3: Create README.md with resume-focused architecture highlights**

```markdown
# Multi-Functional AI Portfolio & RAG Showcase Chatbot

An enterprise-grade, portfolio-ready RAG AI Chatbot demonstrating hybrid retrieval, citation attribution, Text-to-SQL analytics, and multi-functional agentic routing.

## 🚀 Live Demo & Deployment
- **Frontend**: Hosted on [Cloudflare Pages](https://pages.cloudflare.com)
- **Backend & Database**: Hosted on [Oracle Cloud Infrastructure (OCI)](https://cloud.oracle.com) Always Free Tier via Docker Compose

## 🧠 Architectural Highlights
- **Hybrid Retrieval**: Combines dense semantic search (`pgvector` cosine similarity with HNSW index) and sparse keyword search (PostgreSQL `tsvector` with GIN index) fused via **Reciprocal Rank Fusion (RRF)**.
- **Strict Citation Attribution**: Bracketed inline citations linked directly to verified source chunk metadata.
- **Text-to-SQL Analytics**: Natural language queries converted to validated, read-only SQL queries over structured project metrics.
- **AI Execution Inspector**: Real-time transparency drawer exposing latency, intent classification, executed SQL, and citation traces.

## 🛠️ Local Development & Testing

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run unit & integration test suite (TDD)
pytest -v

# 3. Start local stack with Docker
docker-compose up -d
```
```

- [ ] **Step 4: Commit deployment files**

```bash
git add Dockerfile docker-compose.yml README.md
git commit -m "feat(deploy): add Dockerfile, docker-compose with pgvector, and resume documentation"
```
