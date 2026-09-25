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
