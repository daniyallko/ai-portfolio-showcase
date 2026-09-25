# Project Standards & Agent Guidelines: Multi-Functional RAG Chatbot

## 1. Project Goal
Build an enterprise-grade, portfolio-ready RAG (Retrieval-Augmented Generation) AI Chatbot featuring multi-tool agentic routing, hybrid search, citation attribution, and automated evaluation metrics.

---

## 2. Core Architectural Pillars

### A. Advanced Retrieval Pipeline
- **Hybrid Retrieval**: Combine dense semantic search (vector embeddings) with sparse keyword search (BM25) using Reciprocal Rank Fusion (RRF).
- **Re-ranking**: Implement a cross-encoder / reranker (e.g. Cohere or BGE-reranker) to score and filter top-k chunks before feeding into the LLM context.
- **Contextual Chunking**: Maintain document hierarchy, parent-child chunk mapping, and document metadata (source, page number, timestamp).

### B. Multi-Functional Agentic Routing
- **Intent Router**: Direct user queries intelligently:
  - `Document Q&A`: Retrieve and synthesize from vector store with strict citation attribution.
  - `Structured Data (SQL/Pandas)`: Query relational databases or tabular datasets.
  - `Live Web Search`: Fetch real-time info using search/fetch tools when knowledge is outside documents.
  - `Direct Tool Execution`: Math calculations, unit conversions, data transformations.
- **Strict Citations**: Every RAG answer must provide clickable or traceable citations referencing specific chunk IDs, source document names, and page numbers.

### C. Observability, Evaluation & Guardrails
- **Evaluation Framework**: Benchmark retrieval and generation quality using **RAGAS** or **TruLens** (measuring Faithfulness, Answer Relevance, and Context Precision).
- **Hallucination Detection**: Validate answers against retrieved context before outputting.
- **Traced Execution**: Log retrieval latency, token usage, and tool routing decisions.

---

## 3. Engineering & Code Standards

- **Test-Driven Development (TDD)**:
  - Write failing unit tests for chunkers, embedders, retrieval filters, and tool routers before writing implementation code.
  - Maintain mock tests for external LLM API calls to ensure deterministic, fast CI test runs.
- **Typing & Structure**:
  - Full Python type hints (`mypy` compliant) and Pydantic models for request/response validation.
  - Clean separation:
    - `core/`: Config, logging, LLM client abstractions.
    - `ingestion/`: Loaders, chunkers, embedding generators.
    - `retrieval/`: Vector store client, BM25 indexer, reranker, hybrid fusion.
    - `agent/`: Router, tools, memory/session manager.
    - `evaluation/`: Ragas test suites, synthetic test datasets.
    - `api/`: FastAPI endpoints with streaming responses (Server-Sent Events).
    - `ui/`: Modern chat UI (Streamlit or Chainlit).
