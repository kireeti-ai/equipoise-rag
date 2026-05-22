"""
evaluate_retrieval.py
Tests the retrieval and reranking pipeline with specific claims to evaluate quality.
"""

from src.retriever import retrieve
from src.reranker import rerank

CLAIMS_TO_TEST = [
    "Metformin extends human lifespan",
    "Vitamin C cures cancer",
    "Coffee increases cardiovascular risk"
]

def evaluate():
    print("="*60)
    print("RETRIEVAL EVALUATION")
    print("="*60)
    
    for claim in CLAIMS_TO_TEST:
        print(f"\nEvaluating Claim: '{claim}'")
        print("-" * 50)
        
        # 1. Retrieve candidate abstracts
        print("Retrieving candidates (Hybrid Dense + BM25)...")
        candidates = retrieve(claim, method="hybrid", top_k=30)
        print(f"Retrieved {len(candidates)} candidates.")
        
        # 2. Rerank
        print("Reranking with Cross-Encoder...")
        reranked = rerank(claim, candidates, top_k=5)
        
        print("\nTop 5 Results:")
        for i, r in enumerate(reranked):
            print(f"  [{i+1}] {r['title']}")
            print(f"      Rerank Score: {r['rerank_score']:.4f} | Original Rank: {r.get('original_rank', '?')}")
            # print a snippet to see context
            snippet = r['text'][:150].replace('\n', ' ')
            print(f"      {snippet}...")
        print("="*60)

if __name__ == "__main__":
    evaluate()
