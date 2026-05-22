"""
reranker.py
Cross-encoder reranker that reranks retrieved abstracts by true relevance.
Uses a cross-encoder model which reads query and document together -- 
more accurate than embedding similarity alone.
"""

from sentence_transformers import CrossEncoder
from src.config import RERANKER_MODEL

_cross_encoder_model = None


def _get_cross_encoder() -> CrossEncoder:
    """Load the cross-encoder once per process and reuse it."""
    global _cross_encoder_model
    if _cross_encoder_model is None:
        _cross_encoder_model = CrossEncoder(RERANKER_MODEL)
    return _cross_encoder_model


def rerank(query: str, retrieved: list, top_k: int = 5) -> list:
    """
    Reranks a list of retrieved abstracts using a cross-encoder model.
    
    Cross-encoder reads query + document together, giving a more accurate
    relevance score than embedding similarity which encodes them separately.
    
    Args:
        query: the original claim
        retrieved: list of dicts from retriever, each with id/title/text/score
        top_k: number of results to return after reranking
    
    Returns:
        reranked list of abstracts, sorted by cross-encoder score
    """
    if not retrieved:
        return []

    model = _get_cross_encoder()

    # Keep track of original ranks for diagnostics
    for i, r in enumerate(retrieved):
        r["original_rank"] = i + 1

    pairs = [[query, r["title"] + " " + r["text"]] for r in retrieved]
    scores = model.predict(pairs)

    for i, result in enumerate(retrieved):
        result["rerank_score"] = float(scores[i])

    reranked = sorted(retrieved, key=lambda x: x["rerank_score"], reverse=True)
    
    print("\n--- Reranking Diagnostics ---")
    for i, r in enumerate(reranked[:top_k]):
        print(f"Rank {i+1} (was {r.get('original_rank', '?')}): Score {r['rerank_score']:.4f} (Orig: {r.get('score', 0):.4f}) | {r['title'][:60]}...")
    print("-----------------------------\n")

    return reranked[:top_k]


if __name__ == "__main__":
    from src.retriever import retrieve

    claim = "Vitamin D supplementation improves depression symptoms"

    print(f"Claim: {claim}\n")

    retrieved = retrieve(claim, method="bm25")

    print("Before reranking:")
    for i, r in enumerate(retrieved):
        print(f"  [{i+1}] {r['title']} | score: {r['score']:.4f}")

    reranked = rerank(claim, retrieved)

    print("\nAfter reranking:")
    for i, r in enumerate(reranked):
        print(f"  [{i+1}] {r['title']} | rerank score: {r['rerank_score']:.4f}")
