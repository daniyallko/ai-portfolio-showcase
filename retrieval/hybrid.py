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
            SearchCandidate(id=str(row.id), content=row.content, metadata=row.metadata_json, score=float(row.similarity))
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
            SearchCandidate(id=str(row.id), content=row.content, metadata=row.metadata_json, score=float(row.rank))
            for row in sparse_rows
        ]

        fused = reciprocal_rank_fusion(dense_candidates, sparse_candidates, k=settings.RRF_K)
        return fused[:top_k]
