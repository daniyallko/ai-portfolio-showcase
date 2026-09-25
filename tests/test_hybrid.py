import pytest
from retrieval.reranker import reciprocal_rank_fusion, SearchCandidate

def test_rrf_scoring_combines_ranks_correctly():
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
    # chunk-1 is in both (rank 1 in dense, rank 2 in sparse) -> highest RRF score
    assert fused[0].id == "chunk-1"
    expected_score = (1.0 / (60 + 1)) + (1.0 / (60 + 2))
    assert pytest.approx(fused[0].rrf_score, 0.0001) == expected_score
