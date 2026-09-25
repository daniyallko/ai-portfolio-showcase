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
