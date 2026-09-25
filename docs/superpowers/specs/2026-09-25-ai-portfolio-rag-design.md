# Design Specification: Multi-Functional AI Portfolio & RAG Showcase Chatbot

**Date**: 2026-09-25  
**Author**: Daniyal  
**Status**: Approved Design  
**Target Deployment**: Frontend on Cloudflare Pages, Backend & Database on Oracle Cloud Infrastructure (Always Free Tier)

---

## 1. Executive Summary & Goals

### 1.1 Objective
Build an enterprise-grade, portfolio-ready AI chatbot platform designed to showcase advanced AI/LLM engineering capabilities to technical recruiters, hiring managers, and clients. 

The chatbot serves a dual purpose:
1. **Interactive Portfolio Agent**: Answers detailed questions about Daniyal’s professional background, skills, education, and projects with verifiable citations referencing actual documents (resume, project case studies).
2. **Interactive AI Capability Showcase**: Allows visitors to test and observe multiple advanced AI tasks via guided demo cards and open-ended chat:
   - **Hybrid RAG** (Dense vector embeddings + Sparse keyword search + RRF reranking)
   - **Text-to-SQL Analytics** (Natural language querying over a structured PostgreSQL portfolio analytics database)
   - **Visitor Document Ingestion & RAG** (Upload custom PDFs to test the RAG engine)
   - **Live Web Research** (Real-time external knowledge retrieval)
   - **Transparent AI Inspector** (Under-the-hood trace of latency, retrieved chunks, RRF scores, and generated SQL)

---

## 2. System Architecture & Component Design

The platform uses a decoupled, production-standard architecture:

```
┌────────────────────────────────────────────────────────┐
│                   Cloudflare Pages                     │
│     (Modern Web UI: Portfolio Bio, Chat Canvas,        │
│       Showcase Cards, AI Inspector Drawer)             │
└───────────────────────────┬────────────────────────────┘
                            │ HTTPS / SSE Streaming
                            ▼
┌────────────────────────────────────────────────────────┐
│             Oracle Cloud Infrastructure (OCI)          │
│               Docker Compose Environment               │
│                                                        │
│  ┌──────────────────────────────────────────────────┐  │
│  │               FastAPI Backend                    │  │
│  │  - /api/chat (SSE streaming)                     │  │
│  │  - /api/upload (Document ingestion)             │  │
│  │  - /api/sql (Direct analytics queries)           │  │
│  │  - /api/health                                   │  │
│  │                                                  │  │
│  │  ┌────────────────────────────────────────────┐  │  │
│  │  │ LangChain Multi-Functional Agentic Router  │  │  │
│  │  │  ├─ Profile RAG Service                    │  │  │
│  │  │  ├─ Text-to-SQL Service (Safety-Guarded)   │  │  │
│  │  │  ├─ Visitor Doc RAG Service                │  │  │
│  │  │  └─ Live Web Search Service                │  │  │
│  │  └────────────────────────────────────────────┘  │  │
│  └──────────────────────────┬───────────────────────┘  │
│                             │                          │
│  ┌──────────────────────────┴───────────────────────┐  │
│  │          PostgreSQL 16 + pgvector                │  │
│  │  - Structured Tables: projects, metrics, skills  │  │
│  │  - Vector Storage: document_chunks (HNSW index)  │  │
│  │  - Full-Text Search: tsv_content (GIN index)     │  │
│  └──────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────┘
```

### 2.1 Directory Structure
```
ai-project/
├── .gitignore
├── AGENTS.md                  # Project rules and agent standards
├── Dockerfile                 # Backend containerization
├── docker-compose.yml         # Multi-container orchestration (FastAPI + Postgres)
├── requirements.txt           # Python dependencies
├── core/
│   ├── config.py              # Pydantic Settings (API keys, DB URLs, configs)
│   ├── database.py            # PostgreSQL connection pool & SQLAlchemy engine
│   └── logging.py             # Structured JSON logging & execution tracing
├── ingestion/
│   ├── loaders.py             # LangChain PDF & Markdown loaders
│   ├── chunker.py             # Recursive chunker with parent-child metadata
│   └── embedder.py            # Gemini text-embedding-004 wrapper
├── retrieval/
│   ├── vector_store.py        # pgvector operations & HNSW index management
│   ├── hybrid.py              # Hybrid search (dense pgvector + sparse tsvector)
│   └── reranker.py            # Reciprocal Rank Fusion (RRF) algorithm
├── sql/
│   ├── database_schema.py     # Schema definitions for portfolio analytics
│   ├── seed_data.py           # Seed records for projects, metrics, and skills
│   └── sql_agent.py           # LangChain schema-aware SQL generator & validator
├── agent/
│   ├── router.py              # LangChain Intent Classifier & Router
│   ├── tools.py               # Tool definitions (RAG, SQL, Web)
│   └── prompts.py             # System prompts with citation & guardrail rules
├── api/
│   ├── main.py                # FastAPI entrypoint
│   └── routes.py              # Endpoints: /api/chat, /api/upload, /api/health
├── ui/                        # Frontend application (Cloudflare Pages deployable)
│   ├── index.html             # Portfolio landing & chat application shell
│   ├── app.js                 # Chat controller, SSE client, and trace renderer
│   └── styles.css             # Sleek dark/light theme portfolio styling
└── tests/
    ├── test_chunker.py        # Tests for chunk boundaries & metadata
    ├── test_hybrid.py         # Tests for RRF math and ranking
    ├── test_sql_agent.py      # Tests for SQL generation & safety blocks
    └── test_router.py         # Tests for intent routing accuracy
```

---

## 3. Data Models & PostgreSQL Schema

### 3.1 Relational Schema (Portfolio Analytics)
Enables visitors to test Text-to-SQL querying against realistic, structured engineering records:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Projects Table
CREATE TABLE projects (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    category VARCHAR(50) NOT NULL, -- e.g., 'RAG / Generative AI', 'Distributed Systems'
    description TEXT NOT NULL,
    tech_stack TEXT[] NOT NULL,
    github_url VARCHAR(255),
    live_url VARCHAR(255),
    status VARCHAR(30) DEFAULT 'completed',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Project Performance Metrics
CREATE TABLE project_metrics (
    id SERIAL PRIMARY KEY,
    project_id INT REFERENCES projects(id) ON DELETE CASCADE,
    metric_name VARCHAR(100) NOT NULL, -- e.g., 'Latency Reduction', 'Context Precision'
    metric_value NUMERIC(10, 2) NOT NULL,
    unit VARCHAR(30) NOT NULL,          -- e.g., '%', 'ms', 'req/sec'
    impact_description TEXT
);

-- Skills & Proficiency Inventory
CREATE TABLE skills_inventory (
    id SERIAL PRIMARY KEY,
    skill_name VARCHAR(50) NOT NULL,
    category VARCHAR(50) NOT NULL,      -- e.g., 'LLMs & RAG', 'Backend', 'Databases'
    proficiency_level VARCHAR(30) NOT NULL, -- 'Expert', 'Advanced'
    years_experience NUMERIC(3, 1) NOT NULL
);
```

### 3.2 Document & Chunk Schema (Hybrid Search with Citations)
Stores source documents and partitioned chunks with dual indexing for dense and sparse search:

```sql
CREATE TABLE documents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    filename VARCHAR(255) NOT NULL,
    doc_type VARCHAR(50) NOT NULL, -- 'resume', 'case_study', 'visitor_upload'
    file_hash VARCHAR(64) UNIQUE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE document_chunks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INT NOT NULL,
    content TEXT NOT NULL,
    metadata JSONB NOT NULL, -- { source, page_number, section_header, chunk_id }
    embedding vector(768),   -- Gemini text-embedding-004 dimensions
    tsv_content tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
);

-- Indexing
CREATE INDEX idx_chunks_embedding ON document_chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX idx_chunks_tsv ON document_chunks USING gin (tsv_content);
```

---

## 4. Ingestion, Hybrid Retrieval & Citations

### 4.1 Ingestion Flow
1. **Load**: LangChain `PyPDFLoader` extracts text and page numbers from PDFs.
2. **Chunk**: `RecursiveCharacterTextSplitter` chunks text to ~600 tokens with 100-token overlap, preserving sentence structures.
3. **Metadata Enrichment**: Every chunk receives `source`, `page_number`, `chunk_id`, and `document_id`.
4. **Embed**: Gemini `text-embedding-004` generates 768-dimensional dense vectors.

### 4.2 Hybrid Search with Reciprocal Rank Fusion (RRF)
To prevent the blind spots of purely dense vector retrieval (such as exact acronym or symbol searches), the pipeline queries both:
1. **Dense Search**: `ORDER BY embedding <=> query_vector LIMIT 20`
2. **Sparse Search**: `WHERE tsv_content @@ websearch_to_tsquery('english', query) ORDER BY ts_rank_cd(tsv_content, query) DESC LIMIT 20`
3. **RRF Algorithm**:
   $$\text{RRF Score}(d) = \sum_{m \in \{\text{dense}, \text{sparse}\}} \frac{1}{60 + \text{rank}_m(d)}$$
4. Top 4-5 chunks are selected and injected into the LLM context.

### 4.3 Strict Citations
Every synthesized statement references specific chunks via bracketed numbers (`[1]`). The response returns an accompanying list of citation objects containing source filename, page number, and matched excerpt for interactive hover cards in the UI.

---

## 5. Agentic Routing & Tool Suite

### 5.1 Intent Routing
The router classifies incoming prompts into:
- `ABOUT_DANIYAL_RAG`: Inquiries regarding background, work history, or portfolio projects.
- `PORTFOLIO_SQL`: Inquiries regarding metrics, statistics, stack comparisons, or project counts.
- `DOC_UPLOAD_RAG`: Questions targeted at a user-uploaded PDF.
- `LIVE_WEB_SEARCH`: Queries needing current live web data.
- `GENERAL_CHAT`: Greetings or conversational pleasantries.

### 5.2 Text-to-SQL Guardrails
- **Read-Only Enforcement**: Query parser rejects any statements containing non-`SELECT` keywords (`DROP`, `DELETE`, `UPDATE`, `INSERT`, `ALTER`, `TRUNCATE`, `EXEC`).
- **Statement Timeout**: Enforces a 3-second query timeout.
- **Structured Response**: Returns both the LLM natural language answer and the raw SQL + tabular results.

---

## 6. Observability: The "AI Inspector" Drawer

Under each assistant response, visitors can click "Inspect AI Execution" to open a transparency drawer:
- **Intent**: Classified intent and routing decision.
- **Latency**: Total execution time in milliseconds.
- **Retrieved Chunks**: Retrieved chunks, dense rank, sparse rank, and final RRF scores.
- **SQL Trace**: Executed query and row count (if SQL route).
- **Model**: Specific Gemini model version utilized.

---

## 7. Deployment Plan (100% Free Tier)

1. **Frontend**: Cloudflare Pages (free unlimited bandwidth, edge CDN, custom domain).
2. **Backend**: Oracle Cloud Always Free Ampere A1 Compute Instance (up to 4 ARM cores, 24 GB RAM).
3. **Network Connection**: Cloudflare Tunnel (`cloudflared`) on the OCI instance routes traffic directly from the custom domain without opening inbound firewall ports.

---

## 8. Test-Driven Development (TDD) Strategy

Before writing implementation code, tests will be created and validated:
1. `test_chunker.py`: Verifies chunk boundaries, token overlap, and metadata preservation.
2. `test_hybrid.py`: Validates RRF calculation logic against deterministic ranked inputs.
3. `test_sql_agent.py`: Confirms valid SQL generation and verifies that destructive queries are successfully blocked.
4. `test_router.py`: Confirms intent classification accuracy across test query sets.
5. Deterministic mocking for Gemini API calls to ensure lightning-fast, reproducible CI test runs.
